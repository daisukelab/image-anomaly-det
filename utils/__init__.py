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


def to_raw_image(torch_img, uint8=False, denorm=True):
    # transpose channels.
    if len(torch_img.shape) == 4: # batch color image N,C,H,W
        img = torch_img.permute(0, 2, 3, 1)
    elif len(torch_img.shape) == 3: # one color image C,H,W
        img = torch_img.permute(1, 2, 0)
    elif len(torch_img.shape) == 2: # one mono image H,W
        img = torch_img
    else:
        raise ValueError(f'image has wrong shape: {len(torch_img.shape)}')
    # single channel mono image (H,W,1) to be (H,W).
    if img.shape[-1] == 1:
        img = img.view(img.shape[:-1])
    # send to the earth, and denorm.
    img = img.detach().cpu().numpy()
    if denorm:
        img = img * 0.5 + 0.5
    if uint8:
        img = (img * 255).astype(np.uint8)
    return img


def pil_crop(pil_img, crop_size, random_crop=True):
    """Crop PIL image, randomly or from its center."""

    w, h = pil_img.size
    if is_array_like(crop_size):
        crop_size_w, crop_size_h = crop_size
    else:
        crop_size_w, crop_size_h = crop_size, crop_size

    if random_crop:
        x = random.randint(0, np.maximum(0, w - crop_size_w))
        y = random.randint(0, np.maximum(0, h - crop_size_h))
    else:
        x = (w - crop_size_w) // 2
        y = (h - crop_size_h) // 2

    return pil_img.crop((x, y, x+crop_size_w, y+crop_size_h))


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


def plt_tiled_imshow(imgs, titles=None, n_cols=5, axis=True):
    """Plot images in tiled fashion."""

    n_row = (len(imgs) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, 3 * n_row))
    for row in range(n_row):
        for col in range(5):
            i = row * n_cols + col
            if i >= len(imgs): break

            plt.subplot(n_row, n_cols, i+1)
            x = imgs[i]
            if isinstance(torch.Tensor): x = to_raw_image(x)
            if x.shape[-1] == 1: x = x.reshape(x.shape[:-1])
            plt.imshow(x)
            if titles is not None:
                plt.title(titles[i])
            if not axis:
                plt.axis('off')


def number_in_str(text, default=0):
    """Extract leftmost number found in given text in int."""

    g = re.search('\d+', str(text))
    return default if g is None else int(g.group(0))


def preprocess_images(files, to_folder, size=None, mode=None, suffix='.png',
                      pre_crop_rect=None, prepend=True, parent_name=True, verbose=True,
                      skip_creation=False):
    """Preprocess image files and put them in a folder with prepended serial number."""

    to_folder = Path(to_folder)
    ensure_folder(to_folder)
    # get last existing number in to_folder, starting id = that + 1.
    cur_id = max([0] + [number_in_str(f.name) for f in to_folder.glob('*'+suffix)]) + 1
    if skip_creation:
        cur_id = 1
    # (w, h) for resize
    if size is not None:
        w, h = size if is_array_like(size) else (size, size)
    # loop over files
    new_names = []
    for i, f in enumerate(files):
        f = Path(f)
        if not f.is_file():
            raise Exception(f'Sample "{f}" doesn\'t exist.')
        new_file_name = ((f'{cur_id + i}_' if prepend else '') +
                         (f'{f.parent.name}_' if parent_name else '') +
                         f.stem + suffix)
        new_file_name = to_folder/new_file_name
        new_names.append(new_file_name)
        if skip_creation:
            assert new_file_name.exists(), f'{new_file_name} Not Found...'
            continue

        img = Image.open(f)
        if pre_crop_rect is not None:
            img = img.crop(pre_crop_rect)
        if size is not None:
            img = img.resize((w, h))
        if mode is not None:
            img = img.convert(mode)
        img.save(new_file_name)
        if verbose:
            print(f' {f.name} -> {to_folder.name}/{new_file_name.name}' +
                  ('' if pre_crop_rect is None else f'pre_crop({pre_crop_rect}) -> ') +
                  ('' if size is None else f' ({w}, {h})'))
        else:
            print(f' {cur_id + i}', end='')
    print()
    return new_names

