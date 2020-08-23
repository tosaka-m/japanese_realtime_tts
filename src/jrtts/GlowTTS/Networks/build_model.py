#coding:utf-8
import torch
from torch import nn
from .models import FlowGenerator

def build_model(model_params={}):
    model = FlowGenerator(**model_params)

    return model
