import torch
from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import random
import numpy as np
import pandas as pd
from pathlib import Path


class ExtVisionDataset(datasets.VisionDataset):

    def __init__(self, root, num_images, album_tfm, transform,
                 target_transform, load_size, crop_size, random_crop):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        self.album_tfm = album_tfm
        self.num_images = num_images
        self.load_size, self.crop_size = load_size, crop_size
        self.random_crop = random_crop
        self.set_epoch(0)
        self.last_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_rand_seed(self, idx):
        """set random seed to keep albumentations transform
        the same among both normal and anomaly pair"""
        random.seed(self.epoch + idx // 2)

    def apply_tfms_crop_norm(self, img, target, idx=0, pt2include=None):
        # keep the album_tfm do the same among pair
        self.set_rand_seed(idx)

        # augment here
        if self.album_tfm is not None:
            data = {"image": np.array(img)}
            augmented = self.album_tfm(**data)
            img = Image.fromarray(augmented["image"])

        # random crop
        x_r = self.load_size // 2 if pt2include is None else pt2include[0]
        y_r = self.load_size // 2 if pt2include is None else pt2include[1]
        margin = self.crop_size // 4 # concerning transform followed
        if self.random_crop:
            xmax = max(0, min(self.load_size - self.crop_size, x_r - margin))
            x = random.randint(min(max(0, x_r - self.crop_size + margin), xmax), xmax)
            ymax = max(0, min(self.load_size - self.crop_size, y_r - margin))
            y = random.randint(min(max(0, y_r - self.crop_size + margin), ymax), ymax)
        else:
            x = max(0, x_r - (self.crop_size // 2))
            y = max(0, y_r - (self.crop_size // 2))
        img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        assert x <= x_r and x_r <= x + self.crop_size and y <= y_r and y_r <= y + self.crop_size, f'x_r={x_r} x={x} y_r={y_r} y={y}'
        #print(x, y, x + self.crop_size, y + self.crop_size)

        # extra pytorch transforms applied here - supposed NOT to transform spacially.
        if self.transform is not None:
            img = self.transform(img)

        # to tensor & normalize
        img = self.last_tfm(img)

        # just in case...
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def load_image(self, img_file, as_transformed=False):
        img = Image.open(img_file)
        img = img.resize((self.load_size, self.load_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if as_transformed:
            img, _ = self.apply_tfms_crop_norm(img, 0, idx=0, pt2include=None) # 0 is dummy

        return img


class AnomalyTwinDataset(ExtVisionDataset):
    """Anomaly Twin dataset for Anomaly Detection."""

    def __init__(self, root, files, train=True, album_tfm=None, transform=None,
                 target_transform=None, suffix='.png', load_size=224, crop_size=224,
                 width_min=1, width_max=16, length_max=225//5, color=True):

        super().__init__(root, len(files) * 2,
                         album_tfm=album_tfm, transform=transform,
                         target_transform=target_transform,
                         load_size=load_size, crop_size=crop_size, random_crop=True)

        self.suffix = suffix
        self.width_min, self.width_max = width_min, width_max
        self.length_max, self.color = length_max, color
        self.classes = ['normal', 'anomaly']

        root = Path(root)
        # make list of files when it is None
        if files is None:
            files = [str(f) for f in root.glob('**/*'+suffix)]
        files = [str(f).replace(str(root)+'/', '') for f in files]
        # assign labels
        labels = [l for _ in range(len(files)) for l in range(len(self.classes))]
        # double the list of files
        files = [f for ff in files for f in [ff, ff]]
        # keep them as data frame
        self.df = pd.DataFrame({'file': files, 'label': labels})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.set_rand_seed(idx)

        img_file, target = self.df.ix[idx, ['file', 'label']].values

        # load image
        img = self.load_image(f'{self.root}/{img_file}')
        # make it as defect twin if index is odd
        if (idx % 2) != 0:
            img, (x, y, x2, y2) = self.anomaly_twin(img)
        else:
            x, y, x2, y2 = self.random_pick_points(img)
        ano_center = ((x + x2) // 2, (y + y2) // 2)

        img, target = self.apply_tfms_crop_norm(img, target, idx, ano_center)

        return img, target

    def random_pick_point(self, image):
        # randomly choose a point from entire image
        return random.randint(0, self.load_size), random.randint(0, self.load_size)

    def random_pick_points(self, image):
        x, y = self.random_pick_point(image)
        # randomly choose other parameters
        scar_max = self.length_max
        half = self.load_size // 2
        dx, dy = random.randint(0, scar_max), random.randint(0, scar_max)
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
    def __init__(self, root, files, train=True,
                 album_tfm=None, transform=None,
                 target_transform=None, suffix='.png', load_size=224, crop_size=224,
                 width_min=1, width_max=14, length_max=225//5, color=True,
                blob_th=20):

        super().__init__(root, files, train, album_tfm=album_tfm, transform=transform,
                         target_transform=target_transform,
                         suffix=suffix, load_size=load_size, crop_size=crop_size,
                         width_min=width_min, width_max=width_max, 
                         length_max=length_max, color=color)

        self.blob_th = blob_th

    def random_pick_point(self, image):
        # randomly choose a point on object blob
        np_img = np.array(image.filter(ImageFilter.SMOOTH)).astype(np.float32)
        ys, xs = np.where(np.sum(np.abs(np.diff(np_img, axis=0)), axis=2) > self.blob_th)
        x = random.choice(xs)
        ys_x = ys[np.where(xs == x)[0]]
        y = random.randint(ys_x.min(), ys_x.max())
        return x, y

    
class AsIsDataset(ExtVisionDataset):
    """Dataset for load images as is."""

    def __init__(self, root, files, class_labels, album_tfm=None,
                 transform=None, target_transform=None,
                 suffix='.png', load_size=224, crop_size=224, classes=None):

        super().__init__(root, len(files) * 2,
                         album_tfm=album_tfm, transform=transform,
                         target_transform=target_transform,
                         load_size=load_size, crop_size=crop_size, random_crop=False)

        self.suffix, self.load_size = suffix, load_size
        self.classes = sorted(list(set(class_labels))) if classes is None else classes

        root = Path(root)
        # make list of files when it is None
        if files is None:
            files = [str(f) for f in root.glob('**/*'+suffix)]
        files = [str(f).replace(str(root)+'/', '') for f in files]
        # assign labels
        labels = [self.classes.index(l) for l in class_labels]
        # double the list of files
        # keep them as data frame
        self.df = pd.DataFrame({'file': files, 'label': labels})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.set_rand_seed(idx)

        img_file, target = self.df.ix[idx, ['file', 'label']].values
        img = self.load_image(f'{self.root}/{img_file}')
        img, target = self.apply_tfms_crop_norm(img, target, idx, None)

        return img, target
