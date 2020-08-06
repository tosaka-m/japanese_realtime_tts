#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import Optimizer
from .optimizer.radam import RAdam


def build_optimizer(optimizer_parameters):
    optimizers_and_scheduler = {key: _define_optimizer(value) for key, value in optimizer_parameters.items()}
    optimizers = {key: value[0] for key, value in optimizers_and_scheduler.items()}
    schedulers = {key: value[1] for key, value in optimizers_and_scheduler.items()}
    return optimizers, schedulers

def _define_optimizer(params):
    opt_params = params['opt_params']
    sch_params = params['scheduler_params']
    optimizer = AdamW(
        params['params'],
        lr=opt_params.get('lr', 1e-4),
        weight_decay=opt_params.get('weight_decay', 1e-5))
        #amsgrad=True)
    scheduler = _define_scheduler(optimizer, sch_params)
    return optimizer, scheduler

def _define_scheduler(optimizer, params):
    learning_rate_scale_fn = get_learning_rate_fn(
        params.get('step_size', 8e4),
        gamma=params.get('gamma', 0.5),
        warmup_step=params.get('warmup_step', 4000))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_scale_fn)

    return scheduler

def get_learning_rate_fn(step_size, gamma=0.5, warmup_step=4000):

    def _learning_rate_scale_fn(step):
        if step < warmup_step:
            rate = step / warmup_step
        else:
            rate = (gamma) ** ((step - warmup_step) // step_size)

        return rate

    return _learning_rate_scale_fn
