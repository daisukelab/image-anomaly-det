# image-anomaly-det: Image Anomaly Detection

This repository provides image anomaly detection implementation using PyTorch.

## Implemented methods

- atwin: AnomalyTwin
- efficient_gan: EfficientGAN
- dadgt: Deep Anomaly Detection Using Geometric Transformations
- to be continued.

## Setup

### Download/Install

- Grad-CAM: Using pytorch_cnn_visualizations implementation.
- ArcFace: Using arcface_pytorch implementation.
- One cycle policy learning rate scheduler: Using implementation from https://github.com/dkumazaw/onecyclelr.

```sh
pip install -r requirements.txt
git clone https://github.com/daisukelab/pytorch_cnn_visualizations
git clone https://github.com/daisukelab/arcface_pytorch.git
wget https://raw.githubusercontent.com/dkumazaw/onecyclelr/master/onecyclelr.py
```
