#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train Parallel WaveGAN."""

import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")

class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(self,
                 model,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 initial_steps=0,
                 initial_epochs=0,
                 ):
        """Initialize trainer.

        Args:
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.
            initial_steps (int): Initial global steps.
            initial_epochs (int): Initial global epochs.
        """
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.adversarial_step = 1000
        self.adv_weight = 0

        self.model['generator'].to(device)
        self.model['discriminator'].to(device)
        self.criterion['mse'].to(device)
        self.criterion['stft'].to(device)
        if 'pqmf' in self.criterion.keys():
            self.criterion['pqmf'].to(device)

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

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
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            # overwrite schedular argument parameters
            state_dict["scheduler"]["generator"].update(**self.config["generator_scheduler_params"])
            state_dict["scheduler"]["discriminator"].update(**self.config["discriminator_scheduler_params"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        # calculate generator loss
        y_ = self.model["generator"](*x)
        # reconstruct the signal from multi-band signal
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        gen_loss = sc_loss + mag_loss

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            gen_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_train_loss[
                "train/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            self.total_train_loss[
                "train/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        if self.steps > self.config["discriminator_train_start_steps"]:
            self.adversarial_step += 1
            self.adv_weight = self.config["lambda_adv"] * min(1, self.adversarial_step / 1000)
            # keep compatibility
            gen_loss *= self.config.get("lambda_aux_after_introduce_adv_loss", 1.0)
            p_ = self.model["discriminator"](y_)
            if not isinstance(p_, list):
                # for standard discriminator
                adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
            else:
                # for multi-scale discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                adv_loss /= (i + 1)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y)
                    fm_loss = 0.0
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss += self.criterion["l1"](p_[i][j], p[i][j].detach())
                    fm_loss /= (i + 1) * (j + 1)
                    self.total_train_loss["train/feature_matching_loss"] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

            gen_loss += self.adv_weight * adv_loss

        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_train_loss["train/generator_loss"] += gen_loss.item()


        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](*x)
            if self.config["generator_params"]["out_channels"] > 1:
                y_mb_ = y_
                y_ = self.criterion["pqmf"].synthesis(y_mb_)

            # calculate discriminator loss
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())
            if not isinstance(p, list):
                # for standard discriminator
                real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
                fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
                dis_loss = real_loss + fake_loss
                self.total_train_loss["train/real_loss"] += real_loss.item()
                self.total_train_loss["train/fake_loss"] += fake_loss.item()
                self.total_train_loss["train/discriminator_loss"] += dis_loss.item()
            else:
                # for multi-scale discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.criterion["mse"](
                        p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                    fake_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                real_loss /= (i + 1)
                fake_loss /= (i + 1)
                dis_loss = real_loss + fake_loss
                self.total_train_loss["train/real_loss"] += real_loss.item()
                self.total_train_loss["train/fake_loss"] += fake_loss.item()
                self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()
        # update counts
        self.steps += 1


    def _train_epoch(self, data_loader):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(tqdm(data_loader, desc="[train]"), 1):
            # train one step
            self._train_step(batch)

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        self.logger.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")
        train_results = self.total_train_loss.copy()
        self.total_train_loss = defaultdict(float)
        for param_group in self.optimizer['generator'].param_groups:
            print(param_group['lr'])
        for param_group in self.optimizer['discriminator'].param_groups:
            print(param_group['lr'])

        return train_results

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        p_ = self.model["discriminator"](y_)
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        aux_loss = sc_loss + mag_loss

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            aux_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_eval_loss[
                "eval/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            self.total_eval_loss[
                "eval/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        if self.steps > self.config["discriminator_train_start_steps"]:
            # keep compatibility
            aux_loss *= self.config.get("lambda_aux_after_introduce_adv_loss", 1.0)
        if not isinstance(p_, list):
            # for standard discriminator
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            gen_loss = aux_loss + self.adv_weight * adv_loss
        else:
            # for multi-scale discriminator
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
            adv_loss /= (i + 1)
            gen_loss = aux_loss + self.adv_weight * adv_loss

            # feature matching loss
            if self.config["use_feat_match_loss"]:
                p = self.model["discriminator"](y)
                fm_loss = 0.0
                for i in range(len(p_)):
                    for j in range(len(p_[i]) - 1):
                        fm_loss += self.criterion["l1"](p_[i][j], p[i][j])
                fm_loss /= (i + 1) * (j + 1)
                self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
                gen_loss += self.adv_weight * self.config["lambda_feat_match"] * fm_loss

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y)
        p_ = self.model["discriminator"](y_)
        if not isinstance(p_, list):
            # for standard discriminator
            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            dis_loss = real_loss + fake_loss
        else:
            # for multi-scale discriminator
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](
                    p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                fake_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
            real_loss /= (i + 1)
            fake_loss /= (i + 1)
            dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self, data_loader):
        """Evaluate model one epoch."""
        self.logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(data_loader, desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        self.logger.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            self.logger.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # reset
        eval_results = self.total_eval_loss.copy()
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

        return eval_results

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_batch_
            y_batch_ = self.criterion["pqmf"].synthesis(y_mb_)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break
