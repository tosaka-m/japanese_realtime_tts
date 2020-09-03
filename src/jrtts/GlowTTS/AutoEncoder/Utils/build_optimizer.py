#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import Optimizer


class MultiOptimizer:

    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)


    def step(self, key=None):
        if key is not None:
            self.optimizers[key].step()
        else:
            _ = [self.optimizers[key].step() for key in self.keys]

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]

def build_optimizer(parameters_dict, optimizer_params, scheduler_params):
    opt_sch_pairs = {key: _define_optimizer({
        'params': value,
        'optimizer_params': optimizer_params[key],
        'scheduler_params': scheduler_params[key]
    }) for key, value in parameters_dict.items()}
    optimizer = MultiOptimizer(
        optimizers={key: value[0] for key, value in opt_sch_pairs.items()},
        schedulers={key: value[1] for key, value in opt_sch_pairs.items()})
    return optimizer

def _define_optimizer(params):
    optimizer_params = params['optimizer_params']
    sch_params = params['scheduler_params']
    optimizer = AdamW(
        params['params'],
        lr=optimizer_params.get('lr', 1e-4),
        weight_decay=optimizer_params.get('weight_decay', 1e-6),
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = _define_scheduler(optimizer, sch_params)
    return optimizer, scheduler

def _define_scheduler(optimizer, params):
    print(params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params['max_lr'],
        epochs=params['epochs'],
        steps_per_epoch=params['steps_per_epoch'],
        pct_start=0.05,
        final_div_factor=2)

    return scheduler

if __name__=="__main__":
   main()
