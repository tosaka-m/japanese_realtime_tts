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
from Utils.trainer import Trainer
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('-p', '--config_path', default='Configs/base_config.yml', type=str)
@click.option('-t', '--test', is_flag=True)
def main(config_path, test):

    config = yaml.safe_load(open('Configs/base_config.yml'))
    update_config = yaml.safe_load(open(config_path))
    config.update(update_config)
    log_dir = config['outdir']
    if not osp.exists(log_dir): os.mkdir(log_dir)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    if test:
        wandb.init(project="test", config=config)
    else:
        wandb.init(project="ParallelWaveGAN", config=config)
        file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
        logger.addHandler(file_handler)


    batch_size = config['batch_size']
    model_params = config.get('model_params', {})
    train_list, val_list = get_data_path_list()

    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        collate_config=config.get('collate_config', {}))

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                        collate_config=config.get('collate_config', {}))

    generator, discriminator = build_model(
        generator_type=config['generator_type'],
        discriminator_type=config['discriminator_type'],
        model_params={
        'generator': config.get('generator_params', {}),
        'discriminator': config.get('discriminator_params', {})})

    models = {
        'generator': generator,
        'discriminator': discriminator
    }
    optimizer, scheduler = build_optimizer(
        {
            'generator': {'params': models['generator'].parameters(),
                          'opt_params': config.get('generator_optimizer_params', {}),
                          'scheduler_params': config.get('generator_scheduler_params', {})},
            'discriminator': {'params': models['discriminator'].parameters(),
                              'opt_params': config.get('discriminator_optimizer_params', {}),
                              'scheduler_params': config.get('discriminator_scheduler_params', {})},
        }
    )

    critic = build_critic({"stft": config['stft_loss_params'], "out_channels": config['generator_params']['out_channels']})

    device = 'cuda'
    trainer = Trainer(models,
                      criterion=critic,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      config=config,
                      logger=logger,
                      device=device,
                      initial_steps=0,
                      initial_epochs=0)

    trainer.load_checkpoint(config['pretrained_model'], load_only_params=True)
    #trainer.load_checkpoint(config['pretrained_model']) #, load_only_params=True)

    Epoch = config['train_max_steps'] // (len(train_dataloader) // batch_size)
    save_freq = 5
    for epoch in range(1, 1 + Epoch):
        train_results = trainer._train_epoch(train_dataloader)
        logger.info('Epoch: %d  train loss: %.3f' % (epoch, train_results['train/spectral_convergence_loss']))

        val_results = trainer._eval_epoch(val_dataloader)
        logger.info('Epoch: %d  val metric: %.3f' % (epoch, val_results['eval/spectral_convergence_loss']))

        if epoch % save_freq == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%03d.pth' % epoch))

        # upload
        train_results.update(val_results)
        wandb.log(train_results)

    return 0


def get_data_path_list():
    data_list = sorted(glob('Data/jvs/*/*/*/*wav.pth'))
    external_list = sorted(glob('Data/jsut/*/*/*wav.pth'))
    external2_list = sorted(glob('Data/own/*/*wav.pth'))
    external3_list = sorted(glob('Data/predictions/*.pth'))
    print('orig: %d, external: %d, %d' % (len(data_list), len(external_list), len(external2_list)))
    train_list = list(filter(lambda x: re.search(r'jvs\d\d[0-8]', x) is not None, data_list))
    train_list += external_list
    train_list += external2_list
    train_list += external3_list
    val_list = list(filter(lambda x: re.search(r'jvs\d\d[0-8]', x) is None, data_list))

    return train_list, val_list

if __name__=="__main__":
    main()
