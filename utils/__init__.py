import torch
from torch import nn
import numpy as np
from dlcliche.image import show_2D_tSNE
import sys
from pytorch_cnn_visualizations.src.gradcam import GradCam
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
from .options import Options
from .mvtecad import evaluate_MVTecAD


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


def to_norm_image(torch_img):
    return ((torch_img.squeeze(0).numpy()) * 0.5 + 0.5).transpose(1, 2, 0)


def show_heatmap(img, hm, label, alpha=0.5, ax=None, show_original=None):
    """Based on fast.ai implementation..."""
    if ax is None: _, ax = plt.subplots(1, 1)
    ax.set_title(label)
    _im = to_norm_image(img[0])
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


def to_raw_image(img, denorm_tensor=True):
    if type(img) == torch.Tensor:
        # denormalize
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        if denorm_tensor:
            img = (img * 0.5) + 0.5
        img = (img * 255).astype(np.uint8)
    return img


def pil_random_crop(opt, pil_img):
    w, h = pil_img.size
    if opt.random:
        x = random.randint(0, np.maximum(0, w - opt.crop_size))
        y = random.randint(0, np.maximum(0, h - opt.crop_size))
    else:
        x = (w - opt.crop_size) // 2
        y = (h - opt.crop_size) // 2
    return pil_img.crop((x, y, x+opt.crop_size, y+opt.crop_size))


def pil_translate_fill_mirror(img, dx, dy):
    """Translate (shift) image and fill empty part with mirror of the image."""
    w, h = img.size
    four = Image.new(img.mode, (w * 2, h * 2))
    # mirror along x axis
    zy = 0 if dy >= 0 else h
    zxs = [w, 0] if dx >= 0 else [0, w]
    four.paste(img, (zxs[0], zy, zxs[0] + w, zy + h))
    four.paste(img.transpose(Image.FLIP_LEFT_RIGHT), (zxs[1], zy, zxs[1] + w, zy + h))
    # mirror along y axis
    zys = [0, h] if dy >= 0 else [h, 0]
    (four.paste(four.crop((0, zys[0], w * 2, zys[0] + h))
                .transpose(Image.FLIP_TOP_BOTTOM), (0, zys[1], w * 2,  zys[1] + h)))
    # crop translated copy
    zx, zy = int(dx * w), int(dy * h)
    zx, zy = -zx if dx < 0 else w - zx, h + zy if dy < 0 else zy
    return four.crop((zx, zy, zx + w, zy + h))


def plt_tiled_imshow(imgs, titles=None, col_max=5, axis=True):
    n_row = (len(imgs) + col_max - 1) // col_max
    fig = plt.figure(figsize=(15, 3 * n_row))
    for row in range(n_row):
        for col in range(5):
            i = row * col_max + col
            if i >= len(imgs): break

            plt.subplot(n_row, col_max, i+1)
            x = to_raw_image(imgs[i].cpu().detach())
            if x.shape[-1] == 1: x = x.reshape(x.shape[:-1])
            plt.imshow(x)
            if titles is not None:
                plt.title(titles[i])
            if not axis:
                plt.axis('off')


