import os
import os.path as osp

import torch
from .networks import OrigDiscriminator, OrigDiscriminator1d
from .pytorch_unet.unet import UNet

def build_model(model="unet", use_pitch=False):
    if model == "unet":
        model = UNet(n_channels=1, n_classes=1, pitch=use_pitch)
        initialization(model)
        discriminators = [OrigDiscriminator(input_nc=1, n_layers=5),
                          OrigDiscriminator1d(input_nc=80, n_layers=5, ndf=256)]
        _ = [initialization(discriminator) for discriminator in discriminators]
        discriminator = discriminators
    return model, discriminator


def initialization(model):
    for param in model.parameters():
        if len(param.data.shape) <= 1:
            torch.nn.init.uniform_(param, -0.001, 0.001)
        else:
            torch.nn.init.xavier_normal_(param)

if __name__=="__main__":
    build_model()
