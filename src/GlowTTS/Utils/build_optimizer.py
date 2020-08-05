#coding:utf-8
import torch
from torch.optim import AdamW

def build_optimizer(parameters):
    optimizer, scheduler = _define_optimizer(parameters)
    return optimizer, scheduler

def _define_optimizer(params):
    optimizer_params = params['optimizer_params']
    sch_params = params['scheduler_params']
    optimizer = AdamW(
        params['params'],
        lr=optimizer_params.get('lr', 1e-4),
        weight_decay=optimizer_params.get('weight_decay', 0.),
        betas=(0.9, 0.98),
        eps=1e-9)
    #optimizer = RAdam(params['params'])
    scheduler = _define_scheduler(optimizer, sch_params)
    return optimizer, scheduler

def _define_scheduler(optimizer, params):
    print(params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('max_lr', 5e-4),
        epochs=params.get('epochs', 200),
        steps_per_epoch=params.get('steps_per_epoch', 1000),
        pct_start=0.05,
        final_div_factor=200)

    return scheduler
