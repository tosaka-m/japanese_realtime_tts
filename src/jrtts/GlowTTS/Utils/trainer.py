#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train Parallel WaveGAN."""

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib
import seaborn as sns
import numpy as np
import soundfile as sf
import torch
from torch import nn
from PIL import Image

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import matplotlib.pyplot as plt

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(self,
                 model,
                 critic=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.critic = critic
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger

    def _train_epoch(self):
        """Train model one epoch."""
        raise NotImplementedError

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    @torch.no_grad()
    def _get_images(self, **kwargs):
        return {}

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

            # overwrite schedular argument parameters
            state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

class GlowTTSTrainer(Trainer):
    def _train_epoch(self):
        train_losses = defaultdict(list)
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            self.optimizer.zero_grad()
            batch = [b.to(self.device) for b in batch]
            text, text_lengths, mel_target, mel_target_lengths, f0s, speaker_ids = batch
            (z, y_mean, y_logstd, logdet), attn, logw, logw_, x_mean, x_logstd, mean_pitch, x_pitch\
                = self.model(text, text_lengths, mel_target, mel_target_lengths,
                             g=speaker_ids, gen=False,
                             pitch=f0s)
            loss_mle = self.critic['mle'](z, y_mean, y_logstd, logdet, mel_target_lengths)
            loss_length = torch.sum((logw - logw_)**2) / torch.sum(text_lengths)
            loss = loss_mle + loss_length
            if x_pitch is not None:
                loss_pitch = torch.sum((mean_pitch - x_pitch)**2) / torch.sum(text_lengths)
                loss += loss_pitch
                train_losses["train/pitch"].append(loss_pitch.item())

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
            self.optimizer.step()
            self.scheduler.step()
            train_losses["train/mle"].append(loss_mle.item())
            train_losses["train/length"].append(loss_length.item())
            train_losses["train/loss"].append(loss.item())
            total_grad_norm = self.get_gradient_norm(self.model)
            train_losses['train/total_gn'].append(total_grad_norm)
        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device) for b in batch]
            text, text_lengths, mel_target, mel_target_lengths, f0s, speaker_ids = batch
            (z, y_mean, y_logstd, logdet), attn, logw, logw_, x_mean, x_logstd, mean_pitch, x_pitch\
                = self.model(text, text_lengths, mel_target, mel_target_lengths,
                             g=speaker_ids,
                             gen=False,
                             pitch=f0s)
            loss_mle = self.critic['mle'](z, y_mean, y_logstd, logdet, mel_target_lengths)
            loss_length = torch.sum((logw - logw_)**2) / torch.sum(text_lengths)

            loss = loss_mle + loss_length
            if x_pitch is not None:
                loss_pitch = torch.sum((mean_pitch - x_pitch)**2) / torch.sum(text_lengths)
                loss += loss_pitch
                eval_losses["eval/pitch"].append(loss_pitch.item())

            eval_losses["eval/mle"].append(loss_mle.item())
            eval_losses["eval/length"].append(loss_length.item())
            eval_losses["eval/loss"].append(loss.item())
            if eval_steps_per_epoch == 1:
                eval_images["eval/image"].append(self.get_image(
                    [attn[0, 0].cpu().numpy(), mel_target[0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses

    @staticmethod
    def get_image(arrs):
        pil_images = []
        height = 0
        width = 0
        for arr in arrs:
            uint_arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)
            pil_image = Image.fromarray(uint_arr)
            pil_images.append(pil_image)
            height += uint_arr.shape[0]
            width = max(width, uint_arr.shape[1])

        palette = Image.new('L', (width, height))
        curr_heigth = 0
        for pil_image in pil_images:
            palette.paste(pil_image, (0, curr_heigth))
            curr_heigth += pil_image.size[1]

        return palette
