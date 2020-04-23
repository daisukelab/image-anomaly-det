import torch
from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import random
import numpy as np
import pandas as pd
from pathlib import Path


class AnoTwinBaseDataset(datasets.VisionDataset):

    def __init__(self, files, labels, album_tfm, transform,
                 target_transform, load_size, crop_size, online_pre_crop_rect,
                 random_crop, mixup_alpha):
        super().__init__('.', transform=transform,
                         target_transform=target_transform)

        self.album_tfm = album_tfm
        self.load_size, self.crop_size = load_size, crop_size
        self.online_pre_crop_rect = online_pre_crop_rect
        self.random_crop, self.mixup_alpha = random_crop, mixup_alpha
        self.set_epoch(0)
        self.last_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.base_seed = 0

        # keep files/labels in data frame
        self.df = pd.DataFrame({'file': files, 'label': labels})

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_rand_base_seed(self, base_seed):
        self.base_seed = base_seed

    def set_rand_seed(self, idx):
        """set random seed to keep albumentations transform
        the same among both normal and anomaly pair"""
        random.seed(self.base_seed + self.epoch + idx // 2)

    def apply_tfms_crop(self, img, target, idx=0):
        # keep the album_tfm do the same among pair
        #self.set_rand_seed(idx)

        # mixup
        if self.mixup_alpha > 0.0:
            lambd = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            img_file = self.df.sample().file.values[0]
            counter_img = self.load_image(img_file)
            img = np.array(img) * lambd + np.array(counter_img) * (1 - lambd)
            img = Image.fromarray(img.astype(np.uint8))

        # augment here
        if self.album_tfm is not None:
            data = {"image": np.array(img)}
            augmented = self.album_tfm(**data)
            img = Image.fromarray(augmented["image"])

        # random crop
        if self.random_crop:
            x = random.randint(0, max(0, self.load_size - self.crop_size))
            y = random.randint(0, max(0, self.load_size - self.crop_size))
        else:
            x = max(0, (self.load_size - self.crop_size) // 2)
            y = max(0, (self.load_size - self.crop_size) // 2)
        img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

        # extra pytorch transforms applied here - supposed NOT to transform spacially.
        if self.transform is not None:
            img = self.transform(img)

        # just in case...
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def to_tensor_norm(self, img):
        return self.last_tfm(img)

    def load_image(self, img_file, as_transformed=False, seed_idx=0):
        img = Image.open(img_file)
        if self.online_pre_crop_rect:
            img = img.crop(self.online_pre_crop_rect)
        img = img.resize((self.load_size, self.load_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if as_transformed:
            img, _ = self.apply_tfms_crop(img, 0, idx=seed_idx) # 0 is dummy
            img    = self.to_tensor_norm(img)

        return img


class AnomalyTwinDataset(AnoTwinBaseDataset):
    """Anomaly Twin dataset for Anomaly Detection."""

    def __init__(self, files, train=True, album_tfm=None, transform=None,
                 target_transform=None, load_size=224, crop_size=224,
                 width_min=1, width_max=16, length_max=225//5, color=True,
                 online_pre_crop_rect=None, random_crop=True, mixup_alpha=0.0):

        # assign labels; normal, anomaly, normal, anomaly, ...
        self.classes = ['normal', 'anomaly']
        labels = [l for _ in range(len(files)) for l in range(len(self.classes))]
        # double the list of files
        files = [f for ff in files for f in [ff, ff]]

        super().__init__(files, labels,
                         album_tfm=album_tfm, transform=transform,
                         target_transform=target_transform,
                         load_size=load_size, crop_size=crop_size,
                         online_pre_crop_rect=online_pre_crop_rect,
                         random_crop=random_crop,
                         mixup_alpha=mixup_alpha)

        self.width_min, self.width_max = width_min, width_max
        self.length_max, self.color = length_max, color

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.set_rand_seed(idx)

        img_file, target = self.df.loc[idx, ['file', 'label']].values

        # load image
        img = self.load_image(img_file)

        # transform
        img, target = self.apply_tfms_crop(img, target)

        # make it as defect twin if index is odd
        if (idx % 2) != 0:
            img, _ = self.anomaly_twin(img)

        # to tensor, normalize
        img = self.to_tensor_norm(img)

        return img, target

    def random_pick_point(self, image):
        # randomly choose a point from entire image
        return random.randint(0, self.load_size), random.randint(0, self.load_size)

    def random_pick_points(self, image):
        x, y = self.random_pick_point(image)
        # randomly choose other parameters
        half = self.load_size // 2
        dx, dy = random.randint(0, self.length_max), random.randint(0, self.length_max)
        x2, y2 = x + dx if x < half else x - dx, y + dy if y < half else y - dy
        return x, y, x2, y2

    def random_pick_color(self):
        # randomly choose a color
        c = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        if not self.color: c = (c[0], c[0], c[0])
        return c

    def anomaly_twin(self, image):
        """Default anomaly twin maker."""
        # randomly choose a point on object
        x, y, x2, y2 = self.random_pick_points(image)
        # randomly choose other parameters
        c = self.random_pick_color()
        w = random.randint(self.width_min, self.width_max)
        # print(x, y, x2, y2, c, w)
        ImageDraw.Draw(image).line((x, y, x2,y2), fill=c, width=w)
        return image, (x, y, x2, y2)


class DefectOnBlobDataset(AnomalyTwinDataset):
    """Derived from AnomalyTwinDataset class,
    this will draw a scar line on the object blob.

    Effective for images with single object like zoom up photo of a single object
    with single-colored background; Photo of a screw on white background for example.

    Note: Easy algorithm is used to find blob, could catch noises; increase BLOB_TH to avoid that.
    """
    def __init__(self, files, train=True,
                 album_tfm=None, transform=None,
                 target_transform=None, load_size=224, crop_size=224,
                 width_min=1, width_max=14, length_max=225//5, color=True,
                 online_pre_crop_rect=None, blob_th=20,
                 random_crop=True, mixup_alpha=0.0):

        super().__init__(files, train, album_tfm=album_tfm, transform=transform,
                         target_transform=target_transform,
                         load_size=load_size, crop_size=crop_size,
                         online_pre_crop_rect=online_pre_crop_rect,
                         width_min=width_min, width_max=width_max, 
                         length_max=length_max, color=color,
                         random_crop=random_crop, mixup_alpha=mixup_alpha)

        self.blob_th = blob_th

    def random_pick_point(self, image):
        # randomly choose a point on object blob
        np_img = np.array(image.filter(ImageFilter.SMOOTH)).astype(np.float32)
        ys, xs = np.where(np.sum(np.abs(np.diff(np_img, axis=0)), axis=2) > self.blob_th)
        x = random.choice(xs)
        ys_x = ys[np.where(xs == x)[0]]
        y = random.randint(ys_x.min(), ys_x.max())
        return x, y

    
class AsIsDataset(AnoTwinBaseDataset):
    """Dataset for load images as is."""

    def __init__(self, files, class_labels, album_tfm=None,
                 transform=None, target_transform=None,
                 load_size=224, crop_size=224, classes=None, online_pre_crop_rect=None):

        # assign labels
        self.classes = sorted(list(set(class_labels))) if classes is None else classes
        labels = [self.classes.index(l) for l in class_labels]

        super().__init__(files, labels,
                         album_tfm=album_tfm, transform=transform,
                         target_transform=target_transform,
                         load_size=load_size, crop_size=crop_size,
                         online_pre_crop_rect=online_pre_crop_rect,
                         random_crop=False, mixup_alpha=0.0)

    def __getitem__(self, idx):
        self.set_rand_seed(idx)

        img_file, target = self.df.loc[idx, ['file', 'label']].values
        img = self.load_image(img_file)
        img, target = self.apply_tfms_crop(img, target, idx)
        img = self.to_tensor_norm(img)

        return img, target
