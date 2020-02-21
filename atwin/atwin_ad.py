
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, models
from pathlib import Path
import matplotlib.pyplot as plt
from dlcliche.utils import ensure_delete, ensure_folder, get_logger, copy_with_prefix

from utils import get_embeddings, get_head_model
from atwin.dataset import AnomalyTwinDataset, DefectOnBlobDataset, AsIsDataset


class AnoTwinAD:
    """Anomaly Twin Anomaly Detector."""

    def __init__(self, project_name, work_folder, valid_pct=0.2, suffix='.jpg', n_mosts=3,
                 resize=384, size=384, batch_size=16, workers=8,
                 anomaly_twin_dataset=DefectOnBlobDataset, logger=None):
        """Prepare for ATAD data handling.

        - A project folder will be created as: work_folder/project_name
        
        Args:
            anomaly_twin_imagelist:
                Any of AnomalyTwinImageList, DefectOnTheEdgeImageList,
                DefectOnBlobImageList.
        """
        self.project, self.work = project_name, work_folder
        self.valid_pct, self.suffix, self.n_mosts = valid_pct, suffix, n_mosts
        self.resize, self.size, self.batch_size, self.workers = resize, size, batch_size, workers
        self.dataset_cls = anomaly_twin_dataset
        self.root = Path(self.work)/self.project
        self.good, self.test, self.runtime = self.root/'good',  self.root/'test',  self.root/'runtime'
        self.reset_work(delete=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger() if logger is None else logger

    def add_good_samples(self, files, prefix='', symlinks=False):
        """Add good (it is normal as well as training) image files.

        - All the images added will be handled as good (normal) samles later on
          throughout training/test phase.
        - These images are used for calculating distances for test samples.

        Example:
            ABC = Path('path/to/abc')
            add_train_samples(ABC.glob('*.jpg'), prefix='abc_')
        """
        self.logger.debug(f'Adding {len(files)} good samples to {self.good}.')
        copy_with_prefix(files, self.good, prefix, symlinks=symlinks)

    def list_good_samples(self):
        return sorted(self.good.glob(f'*{self.suffix}'))
    
    def reset_work(self, delete=True):
        if delete:
            ensure_delete(self.root)
        ensure_folder(self.root)
        ensure_folder(self.good)
        ensure_folder(self.test)
        ensure_folder(self.runtime)
    
    def reset_test(self):
        self.test_df = None

    def set_test_samples(self, label, files, prefix='', symlinks=False):
        if 'test_df' not in self.__dict__:
            self.test_df = pd.DataFrame({'file': [], 'label': []})
        # copy to test folder.
        for f in files:
            f = Path(f)
            # check files are firmly existing.
            if not f.is_file():
                raise Exception(f'Test sample "{f}" doesn\'t exist.')
        test_prefix = f'{prefix}{str(label)}_'
        copy_with_prefix(files, self.test, test_prefix, symlinks=symlinks)
        # add to test data frame.
        self.test_df = pd.concat([
            self.test_df,
            pd.DataFrame({'file': [self.test/(test_prefix+f.name) for f in files],
                          'label': [label] * len(files)}),
        ])

    def create_datasets(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(self.size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        all_train_files = self.list_good_samples()
        n_train = int(len(all_train_files) * (1 - self.valid_pct))
        file_lists = {x: all_train_files[:n_train] if x is 'train' else all_train_files[n_train:]
                         for x in ['train', 'val']}
        self.ds = {x: AnomalyTwinDataset(self.good, file_lists[x],
                                         load_size=self.resize,
                                         transform=data_transforms[x])
                   for x in ['train', 'val']}
        self.dl = {x: data.DataLoader(self.ds[x], batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.workers)
                   for x in ['train', 'val']}

    def create_ref_dataset(self):
        files = self.list_good_samples()
        self.ds['ref'] = AsIsDataset(self.good, files=files,
                                      class_labels=['good'] * len(files),
                                      load_size=self.resize,
                                      transform=self.ds['val'].transform)
        self.dl['ref'] = data.DataLoader(self.ds['ref'], batch_size=self.batch_size,
                                          shuffle=False, num_workers=self.workers)


    def create_test_dataset(self):
        self.ds['test'] = AsIsDataset(self.test, files=self.test_df.file.values,
                                      class_labels=self.test_df.label.values,
                                      load_size=self.resize,
                                      transform=self.ds['val'].transform)
        self.dl['test'] = data.DataLoader(self.ds['test'], batch_size=self.batch_size,
                                          shuffle=False, num_workers=self.workers)

    def get_test_xx_most_info(self, most_test_idxs, distances, ref_ds, test_ds):
        most_train_idxs = np.argmin(distances[most_test_idxs], axis=1)

        most_train_info = ref_ds.df.iloc[most_train_idxs]
        most_test_info  = test_ds.df.iloc[most_test_idxs]
        #print(distances.shape, most_train_idxs, most_test_idxs)
        most_test_info['distance'] = [distances[_test, _trn]
                                       for _trn, _test in zip(most_train_idxs, most_test_idxs)]
        most_test_info['train_idx'] = most_train_info.index
        most_test_info['train_x'] = most_train_info.file.values
        most_test_info['train_y'] = most_train_info.label.values
        return most_test_info

    def create_model(self, model_weights=None):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        self.model = model_ft.to(self.device)
        self.logger.info(f'Created model.')

        if model_weights is not None:
            self.model.load_state_dict(torch.load(model_weights))
            self.logger.info(f' using model weight: {model_weights}')

    def save_model(self, model_weights):
        torch.save(self.model.state_dict(), model_weights)

    def train_setup(self, model_weights=None):
        self.create_model(model_weights)
        self.create_datasets()

    def runtime_setup(self, model_weights):
        self.create_model(model_weights)
        self.model.eval()
        # create reference
        self.create_datasets()
        self.create_ref_dataset()
        head_model = get_head_model(self.model)
        self.ref_embs = get_embeddings(head_model, self.dl['ref'], self.device)
        self.logger.info('Calculated reference embeddings.')

    def show_test_matching_images(self, title, most_test_info):
        fig, all_axes = plt.subplots(2, self.n_mosts, figsize=(18, 8),
                                     gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(title)
        for j, axes in enumerate(all_axes):
            for i, ax in enumerate(axes):
                cur = most_test_info.loc[most_test_info.index[i]]
                if j == 0:
                    visualize_cnn_by_cam(self.learn, ax=ax, 
                                         label=f'test/{cur.x}\nhas distance={cur.distance:.6f}',
                                         x=pil2tensor(load_rgb_image(self.test/f'{cur.x}')/255,
                                                      np.float32).cuda(), y=0)
                else:
                    show_np_image(load_rgb_image(self.good/cur.train_x), ax=ax)
                    ax.set_title(f'from good {cur.train_x}')

    def __repr__(self):
        def tabbed(text): return text.replace('\n', '\n    ')
        goods = f'# of good samples = {len(self.list_good_samples())}'
        trainset, validset = [f'{x} set:\n {tabbed(str(self.ds[x]) if "ds" in self.__dict__ else "not created yet")}.'
                              for x in ['train', 'val']]
        lines = [goods, trainset, validset]
        return '\n'.join(lines)


