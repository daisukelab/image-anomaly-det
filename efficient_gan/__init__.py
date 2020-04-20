"""
Thanks to a Japanese book 書籍「つくりながら学ぶ! PyTorchによる発展ディープラーニング」（小川雄太郎、マイナビ出版 、19/07/29)

Most of code here is re-assembled from following notebook:
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/6_gan_anomaly_detection/6-4_EfficientGAN.ipynb
"""

import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from dlcliche.utils import (ensure_delete, ensure_folder, get_logger,
                            deterministic_everything, random)
from dlcliche.image import show_np_image, subplot_matrix
from dlcliche.math import n_by_m_distances, np_describe
from sklearn import metrics
from PIL import Image
import time
from utils import maybe_this_or_none, to_raw_image, pil_crop
from base_ano_det import BaseAnoDet


class EfficientGANAnoDet(BaseAnoDet):
    """Anomaly Detector by using EfficientGAN."""

    def __init__(self, params, **kwargs):
        super().__init__(params)
        self.ds, self.dl = {}, {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_model(self, model_weights=None, **kwargs):
        self.G = Generator(z_dim=self.params.z_dim)
        self.D = Discriminator(z_dim=self.params.z_dim)
        self.E = Encoder(z_dim=self.params.z_dim)
        if model_weights is not None:
            self.load_model(model_weights)
        for model in [self.G, self.D, self.E]:
            model.to(self.device)
            model.train()

    def to_train(self):
        for model in [self.G, self.D, self.E]:
            model.train()

    def to_eval(self):
        self.G.train() # G has to be train, quite sensitive to batch norm change.
        self.D.eval()
        self.E.eval()

    def setup_train(self, **kwargs):
        pass

    def _transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def train_model(self, train_samples, **kwargs):
        for phase in ['train', 'val']:
            self.ds[phase] = CropResizeImgDataset(file_list=train_samples,
                load_size=self.params.load_size,
                crop_size=self.params.crop_size,
                transform=self._transform(),
                random=(phase == 'train'),
                online_pre_crop_rect=maybe_this_or_none(self.params.data, 'online_pre_crop_rect'),
                mode='L')
            self.dl[phase] = torch.utils.data.DataLoader(self.ds[phase],
                batch_size=self.params.batch_size, shuffle=(phase == 'train'))

        self.to_train()
        train_efficient_gan(self.device, self.params, self.G, self.D, self.E,
            dataloader=self.dl['train'], lr=self.params.lr,
            num_epochs=self.params.n_epochs)

    def setup_runtime(self, **kwargs):
        pass

    def _model_fn(self, filename, kind):
        f = Path(filename)
        return f.parent/f'{f.stem}_{kind}.pth'

    def save_model(self, model_weights, **kwargs):
        def _save(model, kind):
            f = self._model_fn(model_weights, kind)
            torch.save(model.state_dict(), f)
        _save(self.G, 'G')
        _save(self.D, 'D')
        _save(self.E, 'E')

    def load_model(self, model_weights, **kwargs):
        def _load(model, kind):
            d = torch.load(self._model_fn(model_weights, kind), map_location=self.device)
            model.load_state_dict(d)
        _load(self.G, 'G')
        _load(self.D, 'D')
        _load(self.E, 'E')

    def predict(self, test_samples, test_labels=None, return_raw=False, **kwargs):
        self.ds['test'] = CropResizeImgDataset(file_list=test_samples,
            load_size=self.params.load_size,
            crop_size=self.params.crop_size,
            transform=self._transform(),
            random=False,
            online_pre_crop_rect=maybe_this_or_none(self.params.data, 'online_pre_crop_rect'),
            mode='L')
        self.dl['test'] = torch.utils.data.DataLoader(self.ds['test'],
            batch_size=self.params.batch_size, shuffle=False)

        self.to_eval()
        scores = []
        for x in self.dl['test']:
            x = x.to(self.device)
            z = self.E(x)
            reconstructed = self.G(z)

            _, this_scores, _ = anomaly_score(x, reconstructed, z, 
                self.D, Lambda=self.params.scoring_lambda)
            scores.extend(this_scores.cpu().detach().numpy())
        scores = np.array(scores)

        if return_raw:
            return scores, scores
        return scores


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, z_dim=20):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 8*8*256),
            nn.BatchNorm1d(8*8*256),
            nn.ReLU(inplace=True))

        layers = []
        for k_in, k_out in [(256, 128), (128, 64)]:
            layers.append(nn.ConvTranspose2d(in_channels=k_in, out_channels=k_out,
                               kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(k_out))
            layers.append(nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(*layers)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=1,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh())

        self.apply(init_weights)

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = out.view(z.shape[0], 256, 8, 8)
        out = self.layer3(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):

    def __init__(self, z_dim=20):
        super().__init__()

        self.x_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.x_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.z_layer1 = nn.Linear(z_dim, 512)

        self.last1 = nn.Sequential(
            nn.Linear(16896, 1024), # 16896 = 16384 + 512
            nn.LeakyReLU(0.1, inplace=True))

        self.last2 = nn.Linear(1024, 1)

        self.apply(init_weights)

    def forward(self, x, z):

        x_out = self.x_layer1(x)
        x_out = self.x_layer2(x_out)

        z = z.view(z.shape[0], -1)
        z_out = self.z_layer1(z)

        x_out = x_out.view(z.shape[0], -1)
        out = torch.cat([x_out, z_out], dim=1)
        out = self.last1(out)

        feature = out
        feature = feature.view(feature.size()[0], -1)

        out = self.last2(out)

        return out, feature


class Encoder(nn.Module):

    def __init__(self, z_dim=20):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,
                      stride=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Linear(128 * 16 * 16, z_dim)

        self.apply(init_weights)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(-1, 128 * 16 * 16)
        out = self.last(out)

        return out


class CropResizeImgDataset(data.Dataset):

    def __init__(self, file_list, load_size, crop_size, transform, random, online_pre_crop_rect=None, mode='L'):
        self.file_list = file_list
        self.load_size, self.crop_size = load_size, crop_size
        self.transform = transform
        self.random = random
        self.online_pre_crop_rect = online_pre_crop_rect
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert(self.mode)
        if self.online_pre_crop_rect:
            img = img.crop(self.online_pre_crop_rect)
        img = img.resize((self.load_size, self.load_size))
        img = pil_crop(img, self.crop_size, random_crop=self.random)
        img = self.transform(img)
        return img


def train_efficient_gan(device, params, G, D, E, dataloader, lr, num_epochs):

    lr_ge = lr
    lr_d = lr/4
    beta1, beta2 = 0.5, 0.999
    g_optimizer = torch.optim.AdamW(G.parameters(), lr_ge, [beta1, beta2])
    e_optimizer = torch.optim.AdamW(E.parameters(), lr_ge, [beta1, beta2])
    d_optimizer = torch.optim.AdamW(D.parameters(), lr_d, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    for epoch in range(num_epochs):

        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_e_loss = 0.0
        epoch_d_loss = 0.0

        print('Epoch {}/{}  '.format(epoch, num_epochs), end='')

        for imges in dataloader:

            mini_batch_size = imges.size()[0]
            if mini_batch_size == 1:
                continue

            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            imges = imges.to(device)

            # 1. Train Discriminator
            z_out_real = E(imges)
            d_out_real, _ = D(imges, z_out_real)

            input_z = torch.randn(mini_batch_size, params.z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 2. Train Generator
            input_z = torch.randn(mini_batch_size, params.z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            g_loss = criterion(d_out_fake.view(-1), label_real)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # 3. Train Encoder
            z_out_real = E(imges)
            d_out_real, _ = D(imges, z_out_real)

            e_loss = criterion(d_out_real.view(-1), label_fake)
            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            # loss accumulation
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_e_loss += e_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print(f'D_loss:{epoch_d_loss/batch_size:.4f} G_loss:{epoch_g_loss/batch_size:.4f}'
              f' E_loss:{epoch_e_loss/batch_size:.4f}  {t_epoch_finish - t_epoch_start:.4f} sec.')
        t_epoch_start = time.time()


def visualize_train_and_generated(device, G, train_dataloader, E=None, batch_size=8, z_dim=20):
    """Note: Nothing related between training and generated image. z is random if E is empty."""
    batch_iterator = iter(train_dataloader)
    imgs = next(batch_iterator)

    z = torch.randn(batch_size, z_dim) if E is None else E(imgs.to(device))
    fake_imgs = G(z.to(device))

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(to_raw_image(imgs[i]))
        plt.subplot(2, 5, 5+i+1)
        plt.imshow(to_raw_image(fake_imgs[i]))


def anomaly_score(x, fake_img, z_out_real, D, Lambda=0.1):
    residual_loss = torch.abs(x.mean(1)-fake_img.mean(1))
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    _, x_feature = D(x, z_out_real)
    _, G_feature = D(fake_img, z_out_real)

    discrimination_loss = torch.abs(x_feature-G_feature)
    discrimination_loss = discrimination_loss.view(discrimination_loss.size()[0], -1)
    discrimination_loss = torch.sum(discrimination_loss, dim=1)

    loss_each = (1-Lambda)*residual_loss + Lambda*discrimination_loss

    total_loss = torch.sum(loss_each)

    return total_loss, loss_each, residual_loss
