
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, models
from pathlib import Path
import matplotlib.pyplot as plt
from dlcliche.utils import ensure_delete, ensure_folder, get_logger, copy_with_prefix
from dlcliche.image import show_np_image, subplot_matrix
from dlcliche.math import n_by_m_distances, np_describe
from sklearn import metrics

from utils import (get_embeddings, get_body_model,
                   visualize_cnn_grad_cam, visualize_embeddings,
                   to_np_img)
from atwin.dataset import AnomalyTwinDataset, DefectOnBlobDataset, AsIsDataset
from arcface_pytorch.models import ArcMarginProduct


_backbone_models = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
}

class AnoTwinAD:
    """Anomaly Twin Anomaly Detector."""

    def __init__(self, project_name, work_folder, valid_pct=0.2, suffix='.png', n_mosts=4,
                 resize=384, size=384, batch_size=16, workers=8,
                 model='arc_face', backbone='resnet34', pre_crop_rect=None,
                 train_album_tfm=None, train_tfm=None, anomaly_color=False,
                 dataset_cls=DefectOnBlobDataset, val_ds_cls=AsIsDataset, logger=None):
        """Prepare for ATAD data handling.

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
        self.project, self.work = project_name, work_folder
        self.valid_pct, self.suffix, self.n_mosts = valid_pct, suffix, n_mosts
        self.resize, self.size, self.batch_size, self.workers = resize, size, batch_size, workers
        self.model, self.backbone, self.pre_crop_rect = model, backbone, pre_crop_rect
        self.train_album_tfm, self.train_tfm = train_album_tfm, train_tfm
        self.anomaly_color, self.dataset_cls, self.val_ds_cls = anomaly_color, dataset_cls, val_ds_cls
        self.root = Path(self.work)/self.project
        self.good, self.test = self.root/'good',  self.root/'test'
        self.runtime, self.weights = self.root/'runtime', self.root/'weights'
        self.reset_work(delete=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger() if logger is None else logger

    def get_backbone(self):
        if type(self.backbone) == str and self.backbone in _backbone_models:
            return _backbone_models[self.backbone]
        return self.backbone # expect this option to be a model object

    def add_good_samples(self, files, prefix='', symlinks=False):
        """Add good (it is normal as well as training) image files.

        - All the images added will be handled as good (normal) samles later on
          throughout training/test/runtime phase.
        - These images are used for calculating distances for test samples.

        Example:
            ABC = Path('path/to/abc')
            add_good_samples(ABC.glob('*.jpg'), prefix='abc_')
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
        ensure_folder(self.weights)
    
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
        all_train_files = self.list_good_samples()
        n_train = int(len(all_train_files) * (1 - self.valid_pct))
        file_lists = {x: all_train_files[:n_train] if x is 'train' else all_train_files[n_train:]
                         for x in ['train', 'val']}
        self.ds = {x: self.dataset_cls(self.good, file_lists[x],
                                       load_size=self.resize, crop_size=self.size,
                                       album_tfm=self.train_album_tfm if x == 'train' else None,
                                       transform=self.train_tfm if x == 'train' else None,
                                       pre_crop_rect=self.pre_crop_rect, color=self.anomaly_color)
                   for x in ['train', 'val']}
        self.dl = {x: data.DataLoader(self.ds[x], batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.workers)
                   for x in ['train', 'val']}

    def create_ref_dataset(self):
        files = self.list_good_samples()
        self.ds['ref'] = self.val_ds_cls(self.good, files=files,
                                     class_labels=['good'] * len(files),
                                     load_size=self.resize, crop_size=self.size,
                                     album_tfm=None, transform=None, pre_crop_rect=self.pre_crop_rect)
        self.dl['ref'] = data.DataLoader(self.ds['ref'], batch_size=self.batch_size,
                                         shuffle=False, num_workers=self.workers)


    def create_test_dataset(self):
        self.ds['test'] = self.val_ds_cls(self.test, files=self.test_df.file.values,
                                      class_labels=self.test_df.label.values,
                                      load_size=self.resize, crop_size=self.size,
                                      album_tfm=None, transform=None, pre_crop_rect=self.pre_crop_rect)
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
        base_model = self.get_backbone()(pretrained=True)
        self.model = (get_body_model(base_model)
                      .to(self.device))
        #self.model = nn.DataParallel(self.model)
        num_ftrs = base_model.fc.in_features
        self.metric_fc = (ArcMarginProduct(num_ftrs, 2, s=30, m=0.5, easy_margin=False)
                          .to(self.device))
        #self.metric_fc = nn.DataParallel(self.metric_fc)
        self.logger.info(f'Created model.')

        if model_weights is not None:
            self.load_model(model_weights)
            self.logger.info(f' using model weight: {model_weights}')

    def _model_fn(self, save_name, kind):
        return self.weights/f'{save_name}_{kind}.pth'

    def save_model(self, save_name='trained_weights', weights=None):
        if weights is None:
            weights = {'model': self.model.state_dict(),
                       'metric_fc': self.metric_fc.state_dict()}
        torch.save(weights['model'], self._model_fn(save_name, 'body'))
        torch.save(weights['metric_fc'], self._model_fn(save_name, 'metric-head'))

    def set_model_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.metric_fc.load_state_dict(weights['metric_fc'])

    def load_model(self, save_name='trained_weights'):
        self.model.load_state_dict(torch.load(self._model_fn(save_name, 'body')))
        self.metric_fc.load_state_dict(torch.load(self._model_fn(save_name, 'metric-head')))

    def clf_forward(self, inputs, labels):
        xs = self.model.forward(inputs)
        return self.metric_fc(xs.reshape(xs.size(0), -1), labels)

    def optimizer(self, kind, lr, weight_decay):
        if kind == 'sgd':
            optimizer = torch.optim.SGD([{'params': self.model.parameters()},
                                         {'params': self.metric_fc.parameters()}],
                                        lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                          {'params': self.metric_fc.parameters()}],
                                         lr=lr, weight_decay=weight_decay)
        return optimizer

    def train_setup(self, model_weights=None):
        self.create_model(model_weights)
        self.create_datasets()

    def runtime_setup(self, model_weights):
        self.create_model(model_weights)
        self.model.eval()
        # create reference
        self.create_datasets()
        self.create_ref_dataset()
        self.ref_embs = get_embeddings(self.model, self.dl['ref'], self.device)
        self.logger.info('Calculated reference embeddings.')

    def show_test_matching_images(self, title, most_test_info):
        fig, all_axes = plt.subplots(3, self.n_mosts, figsize=(18, 12),
                                     gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle(title)
        for j, axes in enumerate(all_axes):
            for i, ax in enumerate(axes):
                cur = most_test_info.loc[most_test_info.index[i]]
                if j < 2:
                    visualize_cnn_grad_cam(self.model,
                                           image=(self.ds['test'].load_image(self.test/f'{cur.file}',
                                                                  as_transformed=True)).unsqueeze(0),
                                           title=(f'test/{cur.file}\nhas distance={cur.distance:.6f}'
                                                  if j == 0 else ''),
                                           target_class=j, target_layer=7,
                                           counterfactual=(j == 0),
                                           show_original=('vertical' if j == 0 else None),
                                           ax=ax, separate_head=self.metric_fc, device=self.device)
                else:
                    show_np_image(np.array(self.ds['test'].load_image(self.good/cur.train_x)), ax=ax)
                    ax.set_title(f'from good {cur.train_x}')

    def eval_test(self, vis_class=None, aug_level=0, norm_level=1):
        self.create_test_dataset()
        test_embs, test_y = get_embeddings(self.model, self.dl['test'], self.device, return_y=True)

        distances = n_by_m_distances(test_embs, self.ref_embs)

        test_anomaly_mask = [y != self.ds['test'].classes.index('good') for y in test_y]
        test_anomaly_idx = np.where(test_anomaly_mask)[0]
        y_true = np.array(list(map(int, test_anomaly_mask)))
        preds = np.min(distances, axis=1)
        self.ds['test'].df['distance'] = preds
        self.test_df['distance'] = preds
        #display(self.ds['test'].df)

        # Get worst/best info

        # 1. Get worst case
        preds_y1 = preds[test_anomaly_mask]
        worst_test_idxs = test_anomaly_idx[preds_y1.argsort()[:self.n_mosts]]
        worst_test_info = self.get_test_xx_most_info(worst_test_idxs,
                                                     distances, self.ds['ref'], self.ds['test'])

        # 2. ROC/AUC
        fpr, tpr, self.thresholds = metrics.roc_curve(y_true, preds)
        auc = metrics.auc(fpr, tpr)

        # 3. Get mean_class_distance
        mean_class_distance = [[np.mean(distances[test_y == cur_test_y, :])]
                               for cur_test_y in range(len(self.ds['test'].classes))]
        distance_df = pd.DataFrame(mean_class_distance, columns=[self.project])
        distance_df.index = self.ds['test'].classes

        result = distances, distance_df, (auc, fpr, tpr), worst_test_info
        print(f'AUC = {auc}')

        # Results
        display(distance_df)

        # Embeddings
        labels = self.ds['test'].classes
        good_first_labels = ['good'] + [l for l in labels if l != 'good']
        good_first_map = {labels.index(good_first_labels[i]): i for i in range(len(labels))}
        visualize_embeddings(title='Class embeddings distribution', embeddings=test_embs,
                             ys=map(lambda y: good_first_map[y], test_y),
                             classes=good_first_labels)
        plt.show()

        # Best/Worst cases per class
        if vis_class == -1:
            vis_class = 1 if self.ds['test'].classes[0] == 'good' else 0
        for cls in range(len(self.ds['test'].classes)):
            if vis_class is not None: # None = all
                if self.ds['test'].classes[cls] == 'good': continue
                if vis_class != cls: continue
            test_mask = [y == cls for y in test_y]
            test_idx = np.where(test_mask)[0]
            preds_y1 = preds[test_mask]

            class_worst_test_idxs = test_idx[preds_y1.argsort()[:self.n_mosts]]
            worst_test_info = self.get_test_xx_most_info(class_worst_test_idxs,
                                                         distances, self.ds['ref'], self.ds['test'])
            class_best_test_idxs  = test_idx[preds_y1.argsort()[::-1][:self.n_mosts]]
            best_test_info  = self.get_test_xx_most_info(class_best_test_idxs,
                                                         distances, self.ds['ref'], self.ds['test'])
            if self.ds['test'].classes[cls] == 'good':
                worst_test_info, best_test_info = best_test_info, worst_test_info

            self.show_test_matching_images('Best: ' + self.ds['test'].classes[cls], best_test_info)
            plt.show()

            self.show_test_matching_images('Worst: ' + self.ds['test'].classes[cls], worst_test_info)
            plt.show()

        return result

    def show_samples(self, phase='train', start_index=0, rows=3, columns=3, figsize=(15, 15)):
        for i, ax in enumerate(subplot_matrix(rows=rows, columns=columns, figsize=figsize)):
            cur = start_index + i
            img, label = self.ds[phase][cur]
            img = to_np_img(img)
            ax.imshow(img)
            ax.set_title(f'{cur}:{self.ds[phase].classes[label]}')

    def show_twin_diff(self, phase='train', start_index=0, rows=3, columns=3, figsize=(15, 15)):
        for i, ax in enumerate(subplot_matrix(rows=rows, columns=columns, figsize=figsize)):
            cur = start_index + i * 2
            img1, _ = self.ds[phase][cur]
            img2, _ = self.ds[phase][cur + 1]
            img1, img2 = to_np_img(img1), to_np_img(img2)
            img = np.abs(img2 - img1)
            ax.imshow(img)
            ax.set_title(f'{cur}: {np_describe(img)}')

    def close_up_test_sample(self, img, label, ax=None, figsize=(15, 8)):
        img = img.unsqueeze(0)
        np_img = to_np_img(img)
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
                close_up_test_sample(self, x, disp_label, ax=ax, figsize=figsize)

                if end and end < count:
                    break

    def __repr__(self):
        def tabbed(text): return text.replace('\n', '\n    ')
        goods = f'# of good samples = {len(self.list_good_samples())}'
        trainset, validset = [f'{x} set:\n {tabbed(str(self.ds[x]) if "ds" in self.__dict__ else "not created yet")}.'
                              for x in ['train', 'val']]
        lines = [goods, trainset, validset]
        return '\n'.join(lines)


