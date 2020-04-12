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
                            deterministic_everything, random)
from dlcliche.image import show_np_image, subplot_matrix
from dlcliche.math import n_by_m_distances, np_describe
from sklearn import metrics
from PIL import Image

from arcface_pytorch.models import ArcMarginProduct

from utils import (get_embeddings, get_body_model,
                   visualize_cnn_grad_cam, visualize_embeddings,
                   to_norm_image, set_model_param_trainable)
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


def binary_clf_thresh_by_tpr(thresholds, tpr, min_tpr=1.0):
    """Determine threshold by TPR.
    Example:
        distances, distance_df, (auc, fpr, tpr), worst_test_info = det.eval_test()
        thresh = bin_clf_thresh_by_tpr(det.thresholds, tpr)
    """
    for th, rate in zip(thresholds, tpr):
        if min_tpr <= rate:
            return th
    assert tpr[-1] == 1.0, f'TPR should have 1.0 at the end, why?'
    return None


def binary_clf_thresh_by_fpr(thresholds, fpr, max_fpr=0.1):
    """Determine threshold by FPR.
    Example:
        distances, distance_df, (auc, fpr, tpr), worst_test_info = det.eval_test()
        thresh = det.thresh_by_fpr(det.thresholds, fpr, max_fpr=your_max_fpr)
    """
    for th, next_rate in zip(thresholds, fpr[1:]):
        if max_fpr <= next_rate:
            return th
    # Unfortunately or fortunately all threshold can be under the max_fpr.
    return thresholds[-1]


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

    def __init__(self, params, skip_file_creation=False, **kwargs):
        super().__init__(params=params)
        self.project, self.work = params.project, params.work_folder
        self.suffix, self.n_mosts = params.suffix, params.n_mosts
        self.load_size, self.crop_size = params.load_size, params.crop_size
        self.batch_size, self.workers = params.batch_size, params.workers
        self.model, self.backbone = params.model, params.backbone
        self.train_album_tfm, self.train_tfm = params.train_album_tfm, params.train_tfm
        self.dataset_cls, self.val_ds_cls = params.dataset_cls, params.val_ds_cls

        self.root = Path(self.work)/self.project
        self.good, self.test = self.root/'good', self.root/'test'
        self.runtime, self.weights = self.root/'runtime', self.root/'weights'
        self.reset_work(delete=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger() if params.logger is None else params.logger
        self.train_df = pd.DataFrame()
        self.ds, self.dl = {}, {}
        self.flag_create_files = not skip_file_creation

    def setup_train(self, train_samples, train_set=None, model_weights=None, reset=True):
        # set random seed
        base_seed = self.params.seed if 'seed' in self.params else 0
        base_seed += self.experiment_no
        self.logger.info(f'Random seed: {base_seed}')
        deterministic_everything(base_seed, pytorch=True)

        # prepare files and datasets
        if reset:
            self.reset_work(delete=self.flag_create_files)
            if self.flag_create_files:
                assert len(list(self.test.glob('*'))) == 0, f'{self.test} still have files...'
        self.add_good_samples(train_samples)
        self.create_model(model_weights)
        self.create_datasets(train_set=train_set)

        # follow random seed
        self.ds['train'].set_rand_base_seed(base_seed)
        self.ds['val'].set_rand_base_seed(base_seed)

    def setup_runtime(self, model_weights, ref_samples):
        self.create_model(model_weights)
        self.model.eval()
        self.metric_fc.eval()
        # create reference
        self.add_good_samples(ref_samples)
        self.create_ref_dataset()
        self.ref_embs = get_embeddings(self.model, self.dl['ref'], self.device)
        self.logger.info('Calculated reference embeddings.')

    def get_backbone(self):
        if type(self.backbone) == str and self.backbone in _backbone_models:
            return _backbone_models[self.backbone]
        return self.backbone # expect this option to be a model object

    def get_good_filename(self, f):
        return (self.good/f'{f.parent.name}-{f.name}').with_suffix(self.params.suffix)

    def add_good_samples(self, files):
        """Add good (it is normal as well as training) image files.

        - All the images added will be handled as good (normal) samles later on
          throughout training/test/runtime phase.
        - These images are used for calculating distances for test samples.

        Example:
            ABC = Path('path/to/abc')
            add_good_samples(ABC.glob('*.jpg'))
        """
        self.logger.debug(f'Adding {len(files)} good samples to {self.good}.')
        for f in sorted([str(f) for f in files]):
            f = Path(f)
            if not f.is_file():
                raise Exception(f'Sample "{f}" doesn\'t exist.')
            new_file_name = self.get_good_filename(f)
            if self.flag_create_files:
                Image.open(f).convert('RGB').save(new_file_name)
            self.train_df = pd.concat([
                self.train_df,
                pd.DataFrame({'file': [new_file_name]}),
            ])

    def list_good_samples(self):
        return self.train_df.file.values
    
    def reset_work(self, delete=True, delete_all=False):
        if delete or delete_all:
            ensure_delete(self.good)
            ensure_delete(self.test)
            ensure_delete(self.runtime)
        if delete_all:
            ensure_delete(self.weights)
        ensure_folder(self.root)
        ensure_folder(self.good)
        ensure_folder(self.test)
        ensure_folder(self.runtime)
        ensure_folder(self.weights)
    
    def reset_test(self):
        self.test_df = None

    def set_test_samples(self, files, label='good'):
        if 'test_df' not in self.__dict__:
            self.test_df = pd.DataFrame({'file': [], 'label': []})
        # copy to test folder.
        for f in sorted([str(f) for f in files]):
            f = Path(f)
            if not f.is_file():
                raise Exception(f'Test sample "{f}" doesn\'t exist.')
            new_file_name = ((self.test/f'{f.parent.name}-{f.name}')
                             .with_suffix(self.params.suffix))
            if self.flag_create_files:
                Image.open(f).convert('RGB').save(new_file_name)
            # register file to test set.
            self.test_df = pd.concat([
                self.test_df,
                pd.DataFrame({'file': [new_file_name], 'label': [label]}),
            ])

    def create_datasets(self, train_set=None):
        all_train_files = self.list_good_samples()
        if train_set is None:
            n_train = int(len(all_train_files) * (1 - self.params.valid_pct))
            file_lists = {x: all_train_files[:n_train] if x is 'train' else all_train_files[n_train:]
                           for x in ['train', 'val']}
        else:
            # convert train_set file names to local copy names.
            train_set = [str(self.get_good_filename(f)) for f in train_set]
            # get list of files in each set
            def file_in_train_set(f):
                f = str(f)
                for f_in_ts in train_set:
                    if f_in_ts in f: return True
                return False
            file_lists = {}
            file_lists['train'] = [f for f in all_train_files if file_in_train_set(f)]
            file_lists['val'] = [f for f in all_train_files if not file_in_train_set(f)]
        # duplicate validation set files by n times
        if 'valid_dup_n' in self.params:
            file_lists['val'] = list(file_lists['val']) * self.params.valid_dup_n
        # check data files
        self.logger.debug(f'all train files: {len(all_train_files)}, val files: {len(file_lists["val"])}')
        if len(file_lists["val"]) == 0:
            self.logger.debug('No val files, check train_set')

        self.ds = {x: self.dataset_cls(self.good, file_lists[x],
                                       load_size=self.load_size, crop_size=self.crop_size,
                                       album_tfm=self.train_album_tfm if x == 'train' else None,
                                       transform=self.train_tfm if x == 'train' else None,
                                       width_min=self.params.data.width_min,
                                       width_max=self.params.data.width_max,
                                       length_max=self.params.data.length_max,
                                       color=self.params.data.color,
                                       pre_crop_rect=self.params.data.pre_crop_rect,
                                      )
                   for x in ['train', 'val']}
        self.dl = {x: data.DataLoader(self.ds[x], batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.workers,
                                      worker_init_fn=dataloader_worker_init_fn)
                   for x in ['train', 'val']}

    def create_ref_dataset(self):
        files = self.list_good_samples()
        self.ds['ref'] = self.val_ds_cls(self.good, files=files,
                                         class_labels=['good'] * len(files),
                                         load_size=self.load_size, crop_size=self.crop_size,
                                         album_tfm=None, transform=None,
                                         pre_crop_rect=self.params.data.pre_crop_rect)
        self.dl['ref'] = data.DataLoader(self.ds['ref'], batch_size=self.batch_size,
                                         shuffle=False, num_workers=self.workers)


    def create_test_dataset(self):
        self.ds['test'] = self.val_ds_cls(self.test, files=self.test_df.file.values,
                                          class_labels=self.test_df.label.values,
                                          load_size=self.load_size, crop_size=self.crop_size,
                                          album_tfm=None, transform=None,
                                          pre_crop_rect=self.params.data.pre_crop_rect)
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
        self.metric_fc = (ArcFacePlus(num_ftrs=base_model.fc.in_features)
                          .to(self.device))
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
        result = train_model(self, criterion, optimizer, scheduler, self.dl, num_epochs=10, device=self.device)

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
            scheduler = 'unknown scheduler'
        result = train_model(self, criterion, optimizer, scheduler, self.dl,
                                   num_epochs=self.params.n_epochs, device=self.device)

        if save:
            self.save_model(f'weights_{self.test_target}', weights=result['best_weights'])
        return result

    def predict(self, test_samples):
        self.reset_test()
        self.set_test_samples(test_samples)
        self.create_test_dataset()
        test_embs = get_embeddings(self.model, self.dl['test'], self.device, return_y=False)
        distances = n_by_m_distances(test_embs, self.ref_embs)
        return np.min(distances, axis=1)

    def normalize_distances(self, distances):
        return distances / self.params.distance_norm_factor

    def predict_test(self, test_samples):
        """Perform prediction for evaluation test.
        Note:
            Good samples are supposed to be added, or training is done in advance.
        """
        self.setup_runtime(model_weights=f'weights_{self.test_target}',
                           ref_samples=self.list_good_samples())
        return self.predict(test_samples)

    def draw_heatmap(self, ax, file_name, cls, title='', show_original=None):
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
                    self.draw_heatmap(ax=ax, file_name=self.test/f'{cur.file}',
                        cls=j, title=f'test/{cur.file}\nhas distance={cur.distance:.6f}',
                        show_original=('vertical' if j == 0 else None))
                else:
                    show_np_image(np.array(self.ds['test'].load_image(self.good/cur.train_x)), ax=ax)
                    ax.set_title(f'from good {cur.train_x}')

    def eval_test(self, vis_class=None, aug_level=0, norm_level=1, text_only=False):
        self.create_test_dataset()
        test_embs, test_y = get_embeddings(self.model, self.dl['test'], self.device, return_y=True)

        distances = n_by_m_distances(test_embs, self.ref_embs)

        test_anomaly_mask = [y != self.ds['test'].classes.index('good') for y in test_y]
        test_anomaly_idx = np.where(test_anomaly_mask)[0]
        y_true = np.array(list(map(int, test_anomaly_mask)))

        # 1. Get mean mutual distance
        mean_mutual_class_distance = [[np.mean(distances[test_y == cur_test_y, :])]
                               for cur_test_y in range(len(self.ds['test'].classes))]
        distance_df = pd.DataFrame(mean_mutual_class_distance, columns=[self.project])
        distance_df.index = self.ds['test'].classes
        normalize_factor = max(distance_df.loc['good'].values[0], np.finfo(float).eps)
        self.params.distance_norm_factor = normalize_factor

        # 2. Normalize distances
        distances /= normalize_factor
        distance_df['normalized'] = distance_df[self.project] / normalize_factor

        # Set normalized distances
        preds = np.min(distances, axis=1)
        self.ds['test'].df['distance'] = preds
        self.test_df['distance'] = preds

        # Get worst/best info

        # 3. Get worst case
        preds_y1 = preds[test_anomaly_mask]
        worst_test_idxs = test_anomaly_idx[preds_y1.argsort()[:self.n_mosts]]
        worst_test_info = self.get_test_xx_most_info(worst_test_idxs,
                                                     distances, self.ds['ref'], self.ds['test'])

        # 4. ROC/AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
        auc = metrics.auc(fpr, tpr)

        print(f'AUC = {auc}')

        # Total results
        self.results = (distances, distance_df,
                        (auc, fpr, tpr, thresholds, normalize_factor),
                        worst_test_info)

        if text_only:
            return self.results

        # Visualizations: distances
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

        return self.results

    def calc_test_thresholds(self, max_fpr, min_tpr):
        _, _, (_, fpr, tpr, thresholds, norm_factor), _ = self.results
        return ([2.0, # 2 x mean mutual normalized distance (=normalized 1.0)
                 binary_clf_thresh_by_fpr(thresholds, fpr, max_fpr=max_fpr),
                 binary_clf_thresh_by_tpr(thresholds, tpr, min_tpr=min_tpr)],
                norm_factor)

    def show_samples(self, phase='train', start_index=0, rows=3, columns=3, figsize=(15, 15)):
        for i, ax in enumerate(subplot_matrix(rows=rows, columns=columns, figsize=figsize)):
            cur = start_index + i
            img, label = self.ds[phase][cur]
            img = to_norm_image(img)
            ax.imshow(img)
            ax.set_title(f'{cur}:{self.ds[phase].classes[label]}')

    def show_twin_diff(self, phase='train', start_index=0, rows=3, columns=3, figsize=(15, 15)):
        for i, ax in enumerate(subplot_matrix(rows=rows, columns=columns, figsize=figsize)):
            cur = start_index + i * 2
            img1, _ = self.ds[phase][cur]
            img2, _ = self.ds[phase][cur + 1]
            img1, img2 = to_norm_image(img1), to_norm_image(img2)
            img = np.abs(img2 - img1)
            ax.imshow(img)
            ax.set_title(f'{cur}: {np_describe(img)}')

    def close_up_test_sample(self, img, label, ax=None, figsize=(15, 8)):
        img = img.unsqueeze(0)
        np_img = to_norm_image(img)
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
