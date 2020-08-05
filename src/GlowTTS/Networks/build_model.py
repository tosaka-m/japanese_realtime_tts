#coding:utf-8
import torch
from torch import nn
from .models import FlowGenerator

def build_model(model_params={}):
    model = FlowGenerator(**model_params)
    initialize(model)
    return model

def initialize(model):
    initrange = 0.1
    bias_initrange = 0.001
    parameters = model.parameters()
    for param in parameters:
        if len(param.shape) >= 2:
            torch.nn.init.xavier_normal_(param)
        else:
            torch.nn.init.uniform_(param, (-1)*bias_initrange, bias_initrange)

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-initrange, initrange)
