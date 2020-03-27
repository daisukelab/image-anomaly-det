"""
Class/functions for DADGT implementation.

Implements simplified normality calculation.

- Paper: https://arxiv.org/pdf/1805.10917.pdf
- Thesis: http://www.cs.technion.ac.il/users/wwwb/cgi-bin/tr-get.cgi/2019/MSC/MSC-2019-09.pdf
"""

from dlcliche.utils import *
from dlcliche.math import *

import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
import torchsummary
import pytorch_lightning as pl

from itertools import product
from sklearn import metrics


def to_raw_image(img, denorm_tensor=True):
    if type(img) == torch.Tensor:
        # denormalize
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        if denorm_tensor:
            img = (img * 0.5) + 0.5
        img = (img * 255).astype(np.uint8)
    return img


def random_crop(opt, pil_img):
    w, h = pil_img.size
    if opt.random:
        x = random.randint(0, np.maximum(0, w - opt.crop_size))
        y = random.randint(0, np.maximum(0, h - opt.crop_size))
    else:
        x = (w - opt.crop_size) // 2
        y = (h - opt.crop_size) // 2
    return pil_img.crop((x, y, x+opt.crop_size, y+opt.crop_size))


def translate_fill_mirror(img, dx, dy):
    """Translate (shift) image and fill empty part with mirror of the image."""
    w, h = img.size
    four = Image.new(img.mode, (w * 2, h * 2))
    # mirror along x axis
    zy = 0 if dy >= 0 else h
    zxs = [w, 0] if dx >= 0 else [0, w]
    four.paste(img, (zxs[0], zy, zxs[0] + w, zy + h))
    four.paste(img.transpose(Image.FLIP_LEFT_RIGHT), (zxs[1], zy, zxs[1] + w, zy + h))
    # mirror along y axis
    zys = [0, h] if dy >= 0 else [h, 0]
    (four.paste(four.crop((0, zys[0], w * 2, zys[0] + h))
                .transpose(Image.FLIP_TOP_BOTTOM), (0, zys[1], w * 2,  zys[1] + h)))
    # crop translated copy
    zx, zy = int(dx * w), int(dy * h)
    zx, zy = -zx if dx < 0 else w - zx, h + zy if dy < 0 else zy
    return four.crop((zx, zy, zx + w, zy + h))


def visualize_images(imgs, titles=None, col_max=5, axis=True):
    n_row = (len(imgs) + col_max - 1) // col_max
    fig = plt.figure(figsize=(15, 3 * n_row))
    for row in range(n_row):
        for col in range(5):
            i = row * col_max + col
            if i >= len(imgs): break

            plt.subplot(n_row, col_max, i+1)
            plt.imshow(to_raw_image(imgs[i].cpu().detach()))
            if titles is not None:
                plt.title(titles[i])
            if not axis:
                plt.axis('off')


def create_model(device, n_class, weight_file=None, base_model=models.resnet18):
    model = base_model(pretrained=(weight_file is None))
    model.fc = nn.Linear(model.fc.in_features, n_class)
    model = model.to(device)
    if weight_file is not None:
        model.load_state_dict(torch.load(weight_file))
    return model


class PrepProject(object):
    """Prepare project folders/files."""

    def __init__(self, project_name, train_files, load_size, crop_size, suffix,
                 pre_crop_rect=None, extra='', skip_file_creation=False):
        self.project, self.load_size, self.crop_size = project_name, load_size, crop_size
        self.suffix, self.pre_crop_rect, self.extra = suffix, pre_crop_rect, extra
        self.root = Path(f'./tmp/{self.project}')
        self.train = self.root/'train'
        self.test = self.root/'test'
        self.prepare_train(train_files, skip_file_creation)

    def copy_files_pre_crop(self, dest, files, skip_file_creation):
        new_filenames = []
        for f in files:
            file_name = str((dest/f'{f.parent.name}-{f.name}').with_suffix(self.suffix))
            assert file_name not in new_filenames
            if not skip_file_creation:
                img = Image.open(f).convert('RGB')
                if self.pre_crop_rect is not None:
                    img = img.crop(self.pre_crop_rect)
                img = img.resize((self.load_size, self.load_size))
                img.save(file_name)
            new_filenames.append(file_name)
        return new_filenames

    def prepare_train(self, train_files, skip_file_creation):
        if not skip_file_creation:
            ensure_delete(self.train)
            ensure_folder(self.train)
        self.train_files = self.copy_files_pre_crop(self.train, train_files, skip_file_creation)

    def prepare_test(self, test_files, skip_file_creation=False):
        if not skip_file_creation:
            ensure_delete(self.test)
            ensure_folder(self.test)
        self.test_files = self.copy_files_pre_crop(self.test, test_files, skip_file_creation=skip_file_creation)


class ImageTransform():
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GeoTfmDataset(data.Dataset):
    """Geometric Transformation Dataset.

    Yields:
        image (tensor): Transformed image
        label (int): Transformation label [0, n_tfm() - 1]
    """
    geo_tfms = list(product(
        [None, Image.FLIP_LEFT_RIGHT], 
        [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        [None, [0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]],
    ))

    def __init__(self, file_list, load_size, crop_size, transform, random, debug=None):
        self.file_list = file_list
        self.load_size, self.crop_size = load_size, crop_size
        self.transform, self.random, self.debug = transform, random, debug

    def __len__(self):
        return len(self.file_list) * self.n_tfm()

    def __getitem__(self, index):
        def apply_pil_tanspose(img, tfm_type):
            return img if tfm_type is None else img.transpose(tfm_type)
        def apply_pil_tanslate(img, prms):
            return img if prms is None else translate_fill_mirror(img, prms[0], prms[1])

        if self.debug is not None: print(f'{self.debug}[{index}]')

        img_path = self.filename(index)
        img = Image.open(img_path)
        rot = index % self.n_tfm()
        # resize
        img = img.resize((self.load_size, self.load_size))
        # geometric transform #1: flip
        img = apply_pil_tanspose(img, self.geo_tfms[rot][0])
        # geometric transform #2: rotate
        img = apply_pil_tanspose(img, self.geo_tfms[rot][1])
        # geometric transform #3: translate
        img = apply_pil_tanslate(img, self.geo_tfms[rot][2])
        # crop
        img = random_crop(self, img)
        # transform
        img = self.transform(img)

        return img, rot

    @classmethod
    def n_tfm(cls):
        return len(cls.geo_tfms)

    @classmethod
    def classes(cls):
        return list(range(cls.n_tfm()))

    def filename(self, index):
        file_index = index // self.n_tfm()
        return self.file_list[file_index]


class GeoTfm4Dataset(GeoTfmDataset):
    geo_tfms = list(product(
        [None], 
        [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        [None],
    ))


class GeoTfm20Dataset(GeoTfmDataset):
    geo_tfms = list(product(
        [None], 
        [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        [None, [0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]],
    ))


class GeoTfmEval(object):
    @staticmethod
    def simplified_normality(device, learner, files, n_class, bs_accel=8):
        learner.model.eval()
        ns = []
        for x, _ in learner.get_dataloader(False, files, bs=bs_accel*n_class):
            # predict for a file for all transformations
            ps = learner.model(x.to(device)).softmax(-1)
            ps = ps.detach().cpu().numpy()
            ps = ps.reshape((-1, n_class, n_class))
            # extract predictions for each transformations and take average of them -> normality
            for p in ps:
                n = p.diagonal().mean()
                ns.append(n)
        return ns

    @staticmethod
    def calc(device, learner, files, labels, n_class, bs_accel=1):
        ns = GeoTfmEval.simplified_normality(device, learner, files, n_class, bs_accel=bs_accel)
        fpr, tpr, thresholds = metrics.roc_curve(labels, ns)
        auc = metrics.auc(fpr, tpr)
        return auc, ns


class TrainingScheme(pl.LightningModule):
    """Training scheme by using PyTorch Lightning."""

    def __init__(self, device, model, params, files, ds_cls):
        super().__init__()
        self.device = device
        self.params = params
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        # split data files
        if files is not None:
            n_val = int(params.fit.validation_split * len(files))
            self.val_files = random.sample(files, n_val)
            self.train_files = [f for f in files if f not in self.val_files]
        self.ds_cls = ds_cls

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.fit.lr,
                                betas=(self.params.fit.b1, self.params.fit.b2),
                                weight_decay=self.params.fit.weight_decay)

    def get_dataloader(self, random_shuffle, files, bs=None): 
        ds = self.ds_cls(files,
                          load_size=self.params.data.load_size,
                          crop_size=self.params.data.crop_size,
                          transform=ImageTransform(),
                          random=random_shuffle)
        return torch.utils.data.DataLoader(ds,
                        batch_size=self.params.fit.batch_size if bs is None else bs,
                        shuffle=random_shuffle)

    def train_dataloader(self):
        return self.get_dataloader(True, self.train_files)

    def val_dataloader(self):
        return self.get_dataloader(False, self.val_files)
    
    def save(self, weight_file):
        torch.save(self.model.state_dict(), weight_file)
