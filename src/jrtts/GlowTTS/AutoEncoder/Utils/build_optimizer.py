#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
#from .RAdam.radam import RAdam
import adabound
from .lookahead.lookahead_pytorch import Lookahead

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

def build_optimizer(parameters_dict, lr=1e-4, method="adam", scheduler="step"):
    if not isinstance(lr, dict):
        lr = dict([(k, lr) for k in parameters_dict.keys()])

    optim = dict([(key, _define_optimizer(params, lr[key], method)) \
                   for key, params in parameters_dict.items()])

    schedulers = dict([(key, _define_scheduler(opt, scheduler)) \
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)

    return multi_optim

def _define_optimizer(params, lr=2e-4, method="adam"):
    if method == "adam":
        optimizer = torch.optim.AdamW(params, #RAdam(params,
                                      lr=lr, #amsgrad=True,
                                      weight_decay=1e-6)
        #optimizer = Lookahead(optimizer)
    elif method == "adabound":
        optimizer = adabound.AdaBound(params, #RAdam(params,
                                      lr=lr, final_lr=0.5,
                                      amsbound=True,
                                      weight_decay=1e-6)
    elif method == "sgd":
        print("use sgd")
        optimizer = torch.optim.SGD(params,
                                    lr=lr, momentum=0.9,
                                    weight_decay=1e-6)
    return optimizer

def _define_scheduler(optim, stype="lambda"):

    if stype.lower()=="validation":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=0.5, patience=5, min_lr=5e-5)
    elif stype.lower()=="step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=[50000, 100000, 150000], gamma=0.5)
    elif stype.lower()=="lambda":
        warmup_iteration = 4000
        steps = [100000]
        lr_steps = [1, 0.1]
        def lambda_fn(iteration):
            if iteration < warmup_iteration:
                lr = 1e-7 + lr_steps[0] * (iteration / warmup_iteration)
            elif iteration < steps[0]:
                lr = lr_steps[0] #min_lr + (max_lr - min_lr) * max(0, 1 - iteration / last_iteration)
            else:
                lr = lr_steps[1]

            if iteration in lr_steps:
                print("LR decay", lr)
            return lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda_fn)
    elif stype.lower()=="const":
        def lambda_fn(iteration):
            return 1.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda_fn)

    return scheduler

if __name__=="__main__":
   main()
