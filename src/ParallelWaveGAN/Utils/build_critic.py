#coding:utf-8
import contextlib
import torch
from torch import nn
import torch.nn.functional as F
from .losses.stft_loss import MultiResolutionSTFTLoss
from .losses.pqmf import PQMF

def build_critic(critic_params={}):
    critic = {
        "stft": MultiResolutionSTFTLoss(**critic_params['stft']),
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
    }

    if critic_params.get("out_channels", 1):
        critic["pqmf"] = PQMF(critic_params.get("out_channels"))

    return critic
