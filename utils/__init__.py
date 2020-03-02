import torch
from torch import nn
import numpy as np
from dlcliche.image import show_2D_tSNE
import sys
from pytorch_cnn_visualizations.src.gradcam import GradCam
from skimage.transform import resize
import matplotlib.pyplot as plt
from .options import Options


def get_body_model(model):
    metric_model = nn.Sequential(*list(model.children())[:-1])
    return metric_model


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


def to_np_img(torch_img):
    return ((torch_img.squeeze(0).numpy()) * 0.5 + 0.5).transpose(1, 2, 0)


def show_heatmap(img, hm, label, alpha=0.5, ax=None, show_original=None):
    """Based on fast.ai implementation..."""
    if ax is None: _, ax = plt.subplots(1, 1)
    ax.set_title(label)
    _im = to_np_img(img[0])
    _cm = resize(plt.cm.magma(plt.Normalize()(hm))[:, :, :3], _im.shape)
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
   