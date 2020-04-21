import torch
from torch import nn
import numpy as np
from dlcliche.image import show_2D_tSNE
import sys
from pytorch_cnn_visualizations.src.gradcam import GradCam
import matplotlib.pyplot as plt
import random


def get_body_model(model):
    metric_model = nn.Sequential(*list(model.children())[:-1])
    return metric_model


def set_model_param_trainable(model, flag):
    for param in model.parameters():
        param.requires_grad = flag


def get_embeddings(embedding_model, data_loader, device, return_y=False):
    """Calculate embeddings for all samples in a data_loader.
    
    Args:
        return_y: Also returns labels, for working with training set.
    """
    embedding_model.eval()
    embs, ys = [], []
    for X, y in data_loader:
        #   get embeddings for this batch, store in embs.
        with torch.no_grad():
            # model's output is not softmax'ed.
            out = embedding_model(X.to(device)).cpu().detach().numpy()
            out = out.reshape((len(out), -1))
            embs.append(out)
        ys.append(y.cpu().detach().numpy())
    # Putting all embeddings in shape (number of samples, length of one sample embeddings)
    embs = np.concatenate(embs)
    ys   = np.concatenate(ys)

    return (embs, ys) if return_y else embs


def visualize_embeddings(title, embeddings, ys, classes):
    return show_2D_tSNE(many_dim_vector=embeddings, target=[int(y) for y in ys], title=title, labels=classes)


def show_heatmap(img, hm, label, alpha=0.5, ax=None, show_original=None):
    """Based on fast.ai implementation..."""
    if ax is None: _, ax = plt.subplots(1, 1)
    ax.set_title(label)
    _im = to_raw_image(img[0]) if isinstance(img[0], torch.Tensor) else img[0]
    if hm.shape[:2] != _im.shape[:2]:
        hm = Image.fromarray(hm).resize(_im.shape[1], _im.shape[0])
    _cm = plt.cm.magma(plt.Normalize()(hm))[:, :, :3]
    img = (1 - alpha) * _im + alpha * _cm
    if show_original is not None:
        img = np.concatenate([_im, img], axis=0 if show_original == 'vertical' else 1)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')


def visualize_cnn_grad_cam(model, image, title, target_class=1, target_layer=7, counterfactual=False,
                           ax=None, show_original=None, separate_head=None, device=None):
    grad_cam = GradCam(model, target_layer=target_layer,
                       separate_head=separate_head, device=device)
    # Generate cam mask
    cam = grad_cam.generate_cam(image, target_class, counterfactual=counterfactual)
    #plt.imshow((np_img + cam[..., np.newaxis]) / 2)
    show_heatmap(image, cam, title, ax=ax, show_original=show_original)


def maybe_this_or_none(params, key):
    return params[key] if key in params else None


# --> will be moved to dl-cliche
import re
from pathlib import Path
from dlcliche.utils import ensure_folder, is_array_like
from PIL import Image

from dlcliche.torch_utils import to_raw_image
from dlcliche.image import (pil_crop, pil_translate_fill_mirror, plt_tiled_imshow, preprocess_images)


