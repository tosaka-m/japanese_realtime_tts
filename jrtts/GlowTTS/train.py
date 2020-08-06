#coding:utf-8
import os
import os.path as osp
import re
import sys
import yaml
import shutil
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import wandb
import click

from Utils.build_dataloader import build_dataloader
from Utils.build_optimizer import build_optimizer
from Utils.build_critic import build_critic
from Networks.build_model import build_model
from Utils.trainer import GlowTTSTrainer

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = False

@click.command()
@click.option('-p', '--config_path', default='Configs/base_config.yml', type=str)
@click.option('-t', '--test', is_flag=True)
def main(config_path, test):
    config = yaml.safe_load(open('Configs/base_config.yml'))
    update_config = yaml.safe_load(open(config_path))
    config.update(update_config)

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.mkdir(log_dir)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    if test:
        wandb.init(project="test", config=config)
    else:
        wandb.init(project="glowtts", config=config)
        file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
        logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=6,
                                        dataset_config=config.get('dataset_parasm', {}),
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      dataset_config=config.get('dataset_parasm', {}))

    model = build_model(model_params=config['model_params'] or {})
    model.to(device)
    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 5e-4)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params":{}, "scheduler_params": scheduler_params})


    critic = build_critic(critic_params={'mle': {'n_mels': config['model_params']['out_channels'],
                                                 'n_squeeze': config['model_params']['n_sqz']}})

    trainer = GlowTTSTrainer(model=model,
                             critic=critic,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             device=device,
                             train_dataloader=train_dataloader,
                             val_dataloader=val_dataloader,
                             logger=logger)
    epochs = config.get('epochs', 100)
    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'], load_only_params=False)

    for epoch in range(1, epochs+1):
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info('%-15s: %.4f' % (key, value))
            else:
                results[key] = [wandb.Image(v) for v in value]
        wandb.log(results)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))

    return 0



def get_data_path_list(train_path=None, val_path=None):
    # train_path = "Data/nospace/jsut_train_list.txt"
    # val_path = "Data/nospace/jsut_val_list.txt"
    # train_path = "Data/nospace/train_list.txt"
    # val_path = "Data/nospace/val_list.txt"
    if train_path is None:
        train_path = "Data/nospace/mzk_train_list.txt"
    if val_path is None:
        val_path = "Data/nospace/mzk_val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    # train_list = train_list[:100]
    # val_list = train_list[:100]
    return train_list, val_list

if __name__=="__main__":
    main()
