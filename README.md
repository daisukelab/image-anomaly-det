# image-anomaly-det: Image Anomaly Detection

This repository provides image anomaly detection implementation using PyTorch.

## Implemented methods

- anotwin: AnomalyTwin
- efficient_gan: EfficientGAN
- dadgt: Deep Anomaly Detection Using Geometric Transformations
- img_hash: Image hashing
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

## Example Results

### Image Hashing

| target     |   phash=16 |   phash=32 |   phash=64 |   whash=16 |   whash=32 |
|:-----------|-----------:|-----------:|-----------:|-----------:|-----------:|
| bottle     |   0.894048 |   0.769048 |   0.834921 |   0.831746 |   0.835714 |
| cable      |   0.763681 |   0.71786  |   0.655454 |   0.688999 |   0.756747 |
| capsule    |   0.732948 |   0.764061 |   0.751296 |   0.540088 |   0.713801 |
| carpet     |   0.549559 |   0.476124 |   0.49378  |   0.60012  |   0.461276 |
| grid       |   0.913952 |   0.839181 |   0.786967 |   0.619883 |   0.56391  |
| hazelnut   |   0.78625  |   0.83625  |   0.8125   |   0.462679 |   0.530714 |
| leather    |   0.946671 |   0.897079 |   0.757812 |   0.838655 |   0.793648 |
| metal_nut  |   0.660557 |   0.597996 |   0.556452 |   0.714809 |   0.782014 |
| pill       |   0.649345 |   0.628205 |   0.635706 |   0.648663 |   0.854883 |
| screw      |   0.598176 |   0.686719 |   0.647469 |   0.573786 |   0.605759 |
| tile       |   0.440657 |   0.478896 |   0.450397 |   0.330447 |   0.308983 |
| toothbrush |   0.954167 |   0.969444 |   0.915278 |   0.713889 |   0.806944 |
| transistor |   0.898125 |   0.872708 |   0.814375 |   0.723125 |   0.743958 |
| wood       |   0.502632 |   0.550439 |   0.710088 |   0.393421 |   0.321053 |
| zipper     |   0.737526 |   0.794118 |   0.874081 |   0.463629 |   0.419118 |
