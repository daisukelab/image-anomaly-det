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

sys.path.append('..')
from onecyclelr import OneCycleLR
from base_ano_det import BaseAnoDet
from utils import *


def create_model(device, n_class, weight_file=None, base_model=models.resnet18):
    model = base_model(pretrained=(weight_file is None))
    model.fc = nn.Linear(model.fc.in_features, n_class)
    model = model.to(device)
    if weight_file is not None:
        w = torch.load(weight_file)
        if 'state_dict' in w:
            w = w['state_dict']
        model.load_state_dict(w)
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
                assert not Path(file_name).exists(), f'{file_name} already exists...'
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
            return img if prms is None else pil_translate_fill_mirror(img, prms[0], prms[1])

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
        img = pil_random_crop(self, img)
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
        abnormalities = 1. - np.array(ns)
        fpr, tpr, thresholds = metrics.roc_curve(labels, abnormalities)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.fit.lr,
                                betas=(self.params.fit.b1, self.params.fit.b2),
                                weight_decay=self.params.fit.weight_decay)
        scheduler = OneCycleLR(optimizer, num_steps=self.params.fit.epochs,
                           lr_range=(self.params.fit.lr//10, self.params.fit.lr))
        # ... https://github.com/PyTorchLightning/pytorch-lightning/issues/1120
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        return [optimizer], [scheduler]

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


class DADGT(BaseAnoDet):
    def __init__(self, params, skip_file_creation=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(params=params)
        self.skip_file_creation = skip_file_creation

    def setup_train(self, train_samples):
        deterministic_everything(self.params.seed + self.experiment_no, pytorch=True)
        self.weight_name = f'{self.params.work_folder}/weights-{self.test_target}-'
        #print(' model weight will be stored as:', self.weight_name)

        self.prep =  PrepProject(f'dadgt-{self.params.project}-{self.test_target}', train_samples,
                                 load_size=self.params.data.load_size, crop_size=self.params.data.crop_size,
                                 suffix=self.params.suffix,
                                 skip_file_creation=self.skip_file_creation
                                )

        # visualize to make sure data is fine
        train_dataset = self.params.ds_cls(file_list=self.prep.train_files, load_size=self.prep.load_size,
                                           crop_size=self.prep.crop_size, transform=ImageTransform(), random=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params.fit.batch_size, shuffle=True)
        batch_iterator = iter(train_dataloader)
        imgs, labels = next(batch_iterator)
        print(imgs.size(), labels.size(), len(train_dataset), len(train_dataloader), len(train_dataset.classes()))
        plt_tiled_imshow(imgs[:10], [str(l) for l in labels.detach().cpu().numpy()])
        del train_dataset, train_dataloader

    def train_model(self, train_samples):
        model = create_model(self.device, self.params.n_class, weight_file=None)
        self.learner = TrainingScheme(self.device, model, self. params, self.prep.train_files, self.params.ds_cls)
        chkpt_callback = pl.callbacks.ModelCheckpoint(self.weight_name+'{epoch}-{val_loss:.2f}',
                                                      verbose=True, save_weights_only=True)

        trainer = pl.Trainer(max_epochs=self.params.fit.epochs, gpus=torch.cuda.device_count(),
                             checkpoint_callback=chkpt_callback, show_progress_bar=self.params.fit.show_progress)
        trainer.fit(self.learner)
        self.load_saved_checkpoint()

    def load_saved_checkpoint(self):
        def remove_model(state_dict):
            replaced = {}
            for k in state_dict:
                new_k = k.replace('model.', '') if 'model.' == k[:len('model.')] else k
                replaced[new_k] = state_dict[k]
            return replaced
        path = Path(self.weight_name)
        path = max(path.parent.glob(path.name + '*.ckpt'), key=os.path.getctime)
        print(' loading checkpoint:', path)
        weights = remove_model(torch.load(path)['state_dict'])
        self.learner.model.load_state_dict(weights)

    def predict_test(self, test_samples):
        self.prep.prepare_test(test_samples)
        files = self.prep.test_files
        ns = GeoTfmEval.simplified_normality(self.device, self.learner, files, self.params.n_class, bs_accel=1)
        abnormalities = 1. - np.array(ns)
        return abnormalities

    def load_model(self, weight_file):
        model = create_model(self.device, self.params.n_class, weight_file=weight_file)
        self.learner = TrainingScheme(self.device, model, self.params, None, self.params.ds_cls)
