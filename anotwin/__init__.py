import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, models
from pathlib import Path
import matplotlib.pyplot as plt
from dlcliche.utils import (ensure_delete, ensure_folder, get_logger,
                            deterministic_everything, random, get_class_distribution)
from dlcliche.image import show_np_image, subplot_matrix
from dlcliche.math import n_by_m_distances, np_describe
from sklearn import metrics
from PIL import Image

from arcface_pytorch.models import ArcMarginProduct

from utils import (get_embeddings, get_body_model,
                   visualize_cnn_grad_cam, visualize_embeddings,
                   maybe_this,
                   to_raw_image, set_model_param_trainable)
from anotwin.train import train_model
from anotwin.dataset import AnomalyTwinDataset, DefectOnBlobDataset, AsIsDataset

from base_ano_det import BaseAnoDet


_backbone_models = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
}


class ArcFacePlus(nn.Module):
    def __init__(self, num_ftrs=512, s=30, m=0.5):
        super().__init__()
        self.pre_head = nn.Sequential(
            nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.arc = ArcMarginProduct(512, 2, s=s, m=m, easy_margin=False)

    def forward(self, x, label):
        x = x.view(x.size(0), -1) # Flatten
        x = self.pre_head(x)
        return self.arc(x, label)


def dataloader_worker_init_fn(worker_id):
    random.seed(worker_id)


class AnoTwinDet(BaseAnoDet):
    """Anomaly Twin Detector.

    Working folders will be created as:
    - `work_folder/project_name/good`: storing _good_ samples for training & runtime reference.
    - `work_folder/project_name/test`: storing test samples for evaluation purpose only.
    - `work_folder/project_name/runtime`: TBD, runtime reference samples might be moved here.
    - `work_folder/project_name/weights`: saved model weights will be stored here.

    Args:
        dataset_cls:
            Any of AnomalyTwinImageList, DefectOnTheEdgeImageList,
            DefectOnBlobImageList.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params=params)
        self.project, self.work = params.project, params.work_folder
        self.suffix, self.n_mosts = params.suffix, params.n_mosts
        self.load_size, self.crop_size = params.load_size, params.crop_size
        self.batch_size, self.workers = params.batch_size, params.workers
        self.backbone = params.backbone
        self.train_album_tfm, self.train_tfm = params.train_album_tfm, params.train_tfm
        self.dataset_cls, self.val_ds_cls = params.dataset_cls, params.val_ds_cls

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger = get_logger() if params.logger is None else params.logger
        self.ds, self.dl = {}, {}

    def get_backbone(self):
        if type(self.backbone) == str and self.backbone in _backbone_models:
            return _backbone_models[self.backbone]
        return self.backbone # expect this option to be a model object

    def set_random_seed(self):
        base_seed = self.params.seed if 'seed' in self.params else 0
        if self.experiment_no is not None:
            base_seed += self.experiment_no
        deterministic_everything(base_seed, pytorch=True)
        return base_seed

    def create_model(self, model_weights=None, **kwargs):
        self.set_random_seed()
        base_model = self.get_backbone()(pretrained=True)
        self.model = (get_body_model(base_model)
                      .to(self.device))
        self.metric_fc = (ArcFacePlus(num_ftrs=base_model.fc.in_features)
                          .to(self.device))
        self.logger.info(f'Created model.')
        if model_weights is not None:
            self.load_model(model_weights)
            self.logger.info(f' using model weight: {model_weights}')

    def setup_train(self, train_samples, train_set=None):
        base_seed = self.set_random_seed()
        self.logger.info(f'Random seed: {base_seed}')

        # prepare datasets
        self.create_datasets(train_samples, train_set=train_set)

        # follow random seed
        self.ds['train'].set_rand_base_seed(base_seed)
        self.ds['val'].set_rand_base_seed(base_seed)

    def setup_runtime(self, ref_samples):
        self.model.eval()
        self.metric_fc.eval()
        # create reference
        self.create_ref_dataset(ref_samples)
        self.ref_embs = get_embeddings(self.model, self.dl['ref'], self.device)
        self.logger.info('Calculated reference embeddings.')

    def create_datasets(self, train_samples, train_set=None):
        if train_set is None:
            n_train = int(len(train_samples) * (1 - self.params.valid_pct))
            file_lists = {x: train_samples[:n_train] if x is 'train' else train_samples[n_train:]
                           for x in ['train', 'val']}
        else:
            # convert train_set path names to file names.
            train_set = [Path(f).name for f in train_set]
            # get list of files in each set
            file_lists = {}
            file_lists['train'] = [f for f in train_samples if Path(f).name in train_set]
            file_lists['val'] = [f for f in train_samples if Path(f).name not in train_set]
        # duplicate validation set files by n times
        if 'valid_dup_n' in self.params:
            file_lists['val'] = list(file_lists['val']) * self.params.valid_dup_n
        # check data files
        self.logger.debug(f'all train files: {len(train_samples)}, val files: {len(file_lists["val"])}')
        if len(file_lists["val"]) == 0:
            self.logger.debug('No val files, check train_set')

        self.ds = {}
        self.ds['train'] = self.dataset_cls(file_lists['train'],
                                            load_size=self.load_size, crop_size=self.crop_size,
                                            album_tfm=self.train_album_tfm,
                                            transform=self.train_tfm,
                                            width_min=self.params.data.width_min,
                                            width_max=self.params.data.width_max,
                                            length_max=self.params.data.length_max,
                                            color=self.params.data.color,
                                            online_pre_crop_rect=maybe_this(self.params.data, 'online_pre_crop_rect', None),
                                            random_crop=True,
                                            mixup_alpha=maybe_this(self.params, 'mixup_alpha', 0.0)
                                      )
        self.ds['val'] = self.dataset_cls(file_lists['val'],
                                          load_size=self.load_size, crop_size=self.crop_size,
                                          album_tfm=None,
                                          transform=None,
                                          width_min=self.params.data.width_min,
                                          width_max=self.params.data.width_max,
                                          length_max=self.params.data.length_max,
                                          color=self.params.data.color,
                                          online_pre_crop_rect=maybe_this(self.params.data, 'online_pre_crop_rect', None),
                                          random_crop=maybe_this(self.params.data, 'val_random_crop', False),
                                          mixup_alpha=0.0
                                      )

        self.dl = {x: data.DataLoader(self.ds[x], batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.workers,
                                      worker_init_fn=dataloader_worker_init_fn)
                   for x in ['train', 'val']}

    def create_ref_dataset(self, train_samples):
        self.ds['ref'] = self.val_ds_cls(files=train_samples,
                                         class_labels=['good'] * len(train_samples),
                                         load_size=self.load_size, crop_size=self.crop_size,
                                         album_tfm=None, transform=None,
                                         online_pre_crop_rect=maybe_this(self.params.data, 'online_pre_crop_rect', None))
        self.dl['ref'] = data.DataLoader(self.ds['ref'], batch_size=self.batch_size,
                                         shuffle=False, num_workers=self.workers)

    def create_test_dataset(self, test_samples, test_labels):
        self.ds['test'] = self.val_ds_cls(files=test_samples,
                                          class_labels=test_labels,
                                          load_size=self.load_size, crop_size=self.crop_size,
                                          album_tfm=None, transform=None,
                                          online_pre_crop_rect=maybe_this(self.params.data, 'online_pre_crop_rect', None))
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
        return most_test_info

    def _model_fn(self, save_name, kind):
        return self.weights/f'{save_name}_{kind}.pth'

    def save_model(self, save_name='trained_weights', weights=None, **kwargs):
        if weights is None:
            weights = {'model': self.model.state_dict(),
                       'metric_fc': self.metric_fc.state_dict()}
        torch.save(weights['model'], self._model_fn(save_name, 'body'))
        torch.save(weights['metric_fc'], self._model_fn(save_name, 'metric-head'))

    def load_model(self, save_name='trained_weights', **kwargs):
        d = torch.load(self._model_fn(save_name, 'body'), map_location=self.device)
        self.model.load_state_dict(d)
        d = torch.load(self._model_fn(save_name, 'metric-head'), map_location=self.device)
        self.metric_fc.load_state_dict(d)

    def clf_forward(self, inputs, labels):
        xs = self.model.forward(inputs)
        return self.metric_fc(xs.reshape(xs.size(0), -1), labels)

    def optimizer(self, kind, lr, weight_decay):
        if kind == 'sgd':
            opt = torch.optim.SGD
        elif kind == 'adam':
            opt = torch.optim.Adam
        elif kind == 'adamw':
            opt = torch.optim.AdamW
        else:
            raise Exception('unknown opt')
        optimizer = opt([{'params': self.model.parameters()},
                         {'params': self.metric_fc.parameters()}],
                        lr=lr, weight_decay=weight_decay)
        return optimizer

    def train_model(self, train_samples=None, save=True):
        set_model_param_trainable(self.model, False)
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(kind='sgd', lr=self.params.lr, weight_decay=0.9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=10, max_lr=self.params.lr)
        result = train_model(self, criterion, optimizer, scheduler, self.dl, num_epochs=10,
            flooding_b=maybe_this(self.params, 'flooding_b', 0.0), device=self.device)

        set_model_param_trainable(self.model, True)
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(kind='adamw', lr=self.params.lr/10, weight_decay=self.params.weight_decay)
        if self.params.scheduler == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.params.lr/100, max_lr=self.params.lr,
                            step_size_up=self.params.n_epochs//4, step_size_down=None, cycle_momentum=False,
                            mode='exp_range', gamma=0.98)
        elif self.params.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.params.n_epochs,
                            max_lr=self.params.lr, final_div_factor=100)
        else:
            raise Exception('unknown scheduler')
        result = train_model(self, criterion, optimizer, scheduler, self.dl, num_epochs=self.params.n_epochs,
            flooding_b=maybe_this(self.params, 'flooding_b', 0.0), device=self.device)

        if save:
            self.save_model(f'weights_{self.test_target}', weights=result['best_weights'])
        return result

    def predict(self, test_samples, test_labels=None, return_raw=False):
        self.create_test_dataset(test_samples, ['good']*len(test_samples) if test_labels is None else test_labels)
        test_embs = get_embeddings(self.model, self.dl['test'], self.device, return_y=False)
        sample_distances = n_by_m_distances(test_embs, self.ref_embs)
        if return_raw:
            return sample_distances.min(axis=-1), sample_distances
        return sample_distances.min(axis=-1)

    def draw_heatmap(self, file, distance, kind):
        """Draw heatmap.
        Args:
            file (Path): File path name.
            distance (float): Distance.
            kind (str): 'ok' or 'ng' or anything to describe the file.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.tight_layout()
        for i, ax in enumerate(axes):
            self.draw_heatmap_part(ax=ax, file_name=file, cls=i,
                title=(f'{kind} '+file.name if i == 0 else f'distance={distance:.5f}'))
        return fig

    def draw_heatmap_part(self, ax, file_name, cls, title='', show_original=None):
        """Draw heatmaps on given 2 axes.
        axes: 2 axes.
        cls: Class number: 0 (normal) or 1 (anomaly)
        
        Note:
            Usual Grad-CAM heatmap is used for normal class (0).
            `counterfactual` Grad-CAM heatmap is drawn for anomaly class (1).
        """
        visualize_cnn_grad_cam(self.model,
                            image=(self.ds['test']
                                   .load_image(file_name, as_transformed=True)
                                   .unsqueeze(0)),
                            title=title,
                            target_class=cls, target_layer=7,
                            counterfactual=(cls == 0),
                            show_original=show_original,
                            ax=ax,
                            separate_head=self.metric_fc,
                            device=self.device)

    def show_test_matching_images(self, title, most_test_info):
        fig, all_axes = plt.subplots(3, self.n_mosts, figsize=(18, 12),
                                     gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle(title)
        for j, axes in enumerate(all_axes):
            for i, ax in enumerate(axes):
                cur = most_test_info.loc[most_test_info.index[i]]
                if j < 2:
                    f = Path(cur.file)
                    this_title = (f'{f.parent.name}/{f.name}\nhas distance={cur.distance:.6f}'
                                  if j == 0 else '')
                    self.draw_heatmap_part(ax=ax, file_name=cur.file,
                        cls=j, title=this_title,
                        show_original=('vertical' if j == 0 else None))
                else:
                    f = Path(cur.train_x)
                    show_np_image(np.array(self.ds['test'].load_image(cur.train_x)), ax=ax)
                    ax.set_title(f'from {f.parent.name}/{f.name}')

    def open_image(self, filename):
        """Open as numpy image."""
        return np.array(self.ds['test'].load_image(filename))

    def visualize_after_eval(self, values, test_files, test_labels, test_y_trues):
        auc, pauc, norm_threshs, norm_factor, scores, raw_scores = values

        self.logger.debug(f'# of test files: {len(test_files)}')
        self.logger.debug('distribution' + str(get_class_distribution(test_labels)))
        self.logger.info(f'AUC = {auc}')

        # get worst test info
        test_anomaly_idx = np.where(test_y_trues)[0]
        scores_anomaly = scores[test_anomaly_idx]
        worst_test_idxs = test_anomaly_idx[scores_anomaly.argsort()[:self.n_mosts]]
        worst_test_info = self.get_test_xx_most_info(worst_test_idxs,
                                                    raw_scores, self.ds['ref'], self.ds['test'])

        # visualize embeddings
        classes = sorted(list(set(test_labels)))
        classes = ['good'] + [l for l in classes if l != 'good']
        test_embs = get_embeddings(self.model, self.dl['test'], self.device, return_y=False)
        visualize_embeddings(title='Class embeddings distribution', embeddings=test_embs,
                            ys=[classes.index(label) for label in test_labels],
                            classes=classes)
        plt.show()

        # Best/Worst cases per class
        for cls in classes:
            test_mask = [label == cls for label in test_labels]
            test_idx = np.where(test_mask)[0]
            scores_cls = scores[test_mask]

            class_worst_test_idxs = test_idx[scores_cls.argsort()[:self.n_mosts]]
            worst_test_info = self.get_test_xx_most_info(class_worst_test_idxs,
                                                        raw_scores, self.ds['ref'], self.ds['test'])
            class_best_test_idxs  = test_idx[scores_cls.argsort()[::-1][:self.n_mosts]]
            best_test_info  = self.get_test_xx_most_info(class_best_test_idxs,
                                                        raw_scores, self.ds['ref'], self.ds['test'])
            if cls == 'good':
                worst_test_info, best_test_info = best_test_info, worst_test_info

            self.show_test_matching_images('Best: ' + cls, best_test_info)
            plt.show()

            self.show_test_matching_images('Worst: ' + cls, worst_test_info)
            plt.show()

    def show_samples(self, phase='train', start_index=0, rows=3, columns=3, figsize=(15, 15)):
        for i, ax in enumerate(subplot_matrix(rows=rows, columns=columns, figsize=figsize)):
            cur = start_index + i
            img, label = self.ds[phase][cur]
            img = to_raw_image(img)
            ax.imshow(img)
            ax.set_title(f'{cur}:{self.ds[phase].classes[label]}')

    def show_twin_diff(self, phase='train', start_index=0, rows=3, columns=3, figsize=(15, 15)):
        for i, ax in enumerate(subplot_matrix(rows=rows, columns=columns, figsize=figsize)):
            cur = start_index + i * 2
            img1, _ = self.ds[phase][cur]
            img2, _ = self.ds[phase][cur + 1]
            img1, img2 = to_raw_image(img1), to_raw_image(img2)
            img = np.abs(img2 - img1)
            ax.imshow(img)
            ax.set_title(f'{cur}: {np_describe(img)}')

    def close_up_test_sample(self, img, label, ax=None, figsize=(15, 8)):
        img = img.unsqueeze(0)
        np_img = to_raw_image(img)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        visualize_cnn_grad_cam(self.model.cpu(), img, label, target_class=0, target_layer=7,
                               counterfactual=True, show_original='horizontal',
                               ax=ax, separate_head=self.metric_fc.cpu())

    def close_up_test(self, phase='test', start=None, end=None, ax=None, figsize=(15, 8)):
        count = 0
        for xs, ys in self.dl[phase]:
            for x, y in zip(xs, ys):
                count += 1
                if start and (count-1) < start:
                    continue

                if phase != 'test':
                    y = self.ds['test'].classes.index('good') # force label as 'good'
                label = self.ds['test'].classes[y] # class has to belong to test dataset
                disp_label = f'{phase}[{count - 1}] as class: {label}'
                self.close_up_test_sample(x, disp_label, ax=ax, figsize=figsize)

                if end and end < count:
                    break

    def __repr__(self):
        def tabbed(text): return text.replace('\n', '\n    ')
        trainset, validset = [f'{x} set:\n {tabbed(str(self.ds[x]) if "ds" in self.__dict__ else "not created yet")}.'
                              for x in ['train', 'val']]
        lines = [trainset, validset]
        return '\n'.join(lines)
