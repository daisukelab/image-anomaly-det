import torch
from torchvision import datasets
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import random
import numpy as np
import pandas as pd
from pathlib import Path

class AnomalyTwinDataset(datasets.VisionDataset):
    """Anomaly Twin dataset for Anomaly Detection."""

    def __init__(self, root, files, train=True, transform=None,
                 target_transform=None, suffix='.png', load_size=224,
                 width_min=1, width_max=16, length_max=225//5, color=True):

        super().__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.suffix, self.load_size = suffix, load_size
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
        img_file, target = self.df.ix[idx, ['file', 'label']].values

        # load image
        img = Image.open(f'{self.root}/{img_file}')
        img = img.resize((self.load_size, self.load_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # make it as defect twin if index is odd
        if (idx % 2) != 0:
            img = self.anomaly_twin(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def random_pick_point(self, image):
        # randomly choose a point from entire image
        return random.randint(0, self.load_size), random.randint(0, self.load_size)

    def anomaly_twin(self, image):
        """Default anomaly twin maker."""
        scar_max = self.length_max
        half = self.load_size // 2
        # randomly choose a point on object
        x, y = self.random_pick_point(image)
        # randomly choose other parameters
        dx, dy = random.randint(0, scar_max), random.randint(0, scar_max)
        x2, y2 = x + dx if x < half else x - dx, y + dy if y < half else y - dy
        c = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        if not self.color: c = (c[0], c[0], c[0])
        w = random.randint(self.width_min, self.width_max)
        ImageDraw.Draw(image).line((x, y, x2,y2), fill=c, width=w)
        return image


class DefectOnBlobDataset(AnomalyTwinDataset):
    """Derived from AnomalyTwinImageList class,
    this will draw a scar line on the object blob.

    Effective for images with single object like zoom up photo of a single object
    with single-colored background; Photo of a screw on white background for example.

    Note: Easy algorithm is used to find blob, could catch noises; increase BLOB_TH to avoid that.
    """
    def __init__(self, root, files, train=True, transform=None,
                 target_transform=None, suffix='.png', load_size=224,
                 width_min=1, width_max=14, length_max=225//5, color=True,
                blob_th=20):

        super().__init__(root, files, train, transform=transform,
                         target_transform=target_transform,
                         suffix=suffix, load_size=load_size,
                         width_min=width_min, width_max=width_max, 
                         ength_max=length_max, color=color,
                        )

        self.blob_th = blob_th

    def random_pick_point(self, image):
        # randomly choose a point on object blob
        np_img = np.array(image.filter(ImageFilter.SMOOTH)).astype(np.float32)
        ys, xs = np.where(np.sum(np.abs(np.diff(np_img, axis=0)), axis=2) > self.BLOB_TH)
        x = random.choice(xs)
        ys_x = ys[np.where(xs == x)[0]]
        y = random.randint(ys_x.min(), ys_x.max())
        return x, y

    
class AsIsDataset(datasets.VisionDataset):
    """Anomaly Twin dataset for Anomaly Detection."""

    def __init__(self, root, files, class_labels, transform=None,
                 target_transform=None, suffix='.png', load_size=224, classes=None):

        super().__init__(root, transform=transform,
                                      target_transform=target_transform)

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
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if idx >= len(self.df):
            return None
        img_file, target = self.df.ix[idx, ['file', 'label']].values

        # load image
        img = Image.open(f'{self.root}/{img_file}')
        img = img.resize((self.load_size, self.load_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
