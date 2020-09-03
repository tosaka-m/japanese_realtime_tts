#coding:utf-8
import os, sys
import gc
import os.path as osp
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn
import time
import copy
from collections import defaultdict
from tqdm import tqdm

class AutoEncoderTrainer(object):
    def __init__(self, model, optimizer=None, critic=None):
        self.model = model
        self.optimizer = optimizer
        self.critic = critic
        self.use_cuda=True
        self.clip = 2

    def train(self, dataloader):
        losses = []
        elementwise_losses = []
        print("total data :", len(dataloader.dataset))
        st = time.time()
        self.model.train()

        for idx, data in enumerate(dataloader):
            lq = data["LQ"]
            gt = data["GT"]

            if self.use_cuda:
                lq = lq.cuda()
                gt = [t.cuda() for t in gt]

            pred = self.model(lq)
            loss = self.critic(pred, gt)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.optimizer.scheduler()

            reduced_loss = loss.item()
            losses.append(reduced_loss)
            if (idx+1) % 100 == 0:
                ed = time.time()
                print("iter time: %d : %.3f" % (idx, (ed - st)/60))
                print("loss :", np.mean(losses[-100:]), flush=True)
            torch.cuda.empty_cache()
            gc.collect()

    def validation(self, dataloader, ):
        losses = []
        print("total data :", len(dataloader.dataset))
        self.model.eval()
        for idx, data in enumerate(dataloader):
            lq = data["LQ"]
            gt = data["GT"]
            lq, gt = lq.to(self.device), gt.to(self.device)

            with torch.no_grad():
                pred = self.model(lq)
                loss = self.critic(pred, gt)

            reduced_loss = loss.item()
            losses.append(reduced_loss)

        print("Valid Loss: %.4f" % np.mean(losses))
        return losses


    def save(self, save_dir):
        torch.save(self.model.state_dict(), save_dir + "model.pth")
        torch.save(self.optimizer.state_dict(), save_dir + "optimizer.pth")

    def load(self, model_dir, optimizer=True):
        print("load model")
        state_dict = torch.load(model_dir + "model.pth")
        self._load(state_dict, self.model)

        if self.optimizer is not None and optimizer:
            self.optimizer.load_state_dict(torch.load(model_dir + "optimizer.pth"))

    def _load(self, states, model):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data
                model_states[key].copy_(val)
            except:
                print("not exist ", key)


class SEGANTrainer(object):
    def __init__(self, generator, discriminator=None, optimizer=None, critic=None, device='cpu', adv_weight=0.2, use_pitch=False):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.critic = critic
        self.adv_weight = adv_weight
        self.gen_train_freq = 1
        self.clip = 5
        self.use_pitch = use_pitch
        self.device = device

    def train(self, dataloader):
        losses = []
        st = time.time()
        self.generator.train()
        [d.train() for d in self.discriminator]
        train_losses = defaultdict(list)

        for idx, data in enumerate(tqdm(dataloader, desc="[train]"), 1):

            lq = data["LQ"].to(self.device)
            gt = data["GT"].to(self.device)

            # generator
            if self.use_pitch:
                pitch = data["Pitch"].to(self.device)
                pred = self.generator(lq, pitch)
            else:
                pred = self.generator(lq)

            probs = [discriminator(pred).squeeze(1) \
                    for discriminator in self.discriminator]

            zero_labels = [torch.zeros(p.shape).type_as(p) for p in probs]
            one_labels = [torch.ones(p.shape).type_as(p) for p in probs]

            l1_loss = self.critic["L1"](pred, gt)
            gan_loss = sum([self.critic["L2"](prob, zero_label) \
                            for prob, zero_label in zip(probs, zero_labels)])

            gen_loss = l1_loss + self.adv_weight * gan_loss
            self.optimizer.zero_grad()
            gen_loss.backward()
            # clip grad norm
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip)
            self.optimizer.step("generator")
            reduced_l1_loss = l1_loss.item()
            reduced_gan_loss = gan_loss.item()

            # discrminator
            self.optimizer.zero_grad()

            gt_probs = [discriminator(gt).squeeze(1) for discriminator in self.discriminator]
            gt_loss = sum([self.critic["L2"](gt_prob, zero_label) \
                           for gt_prob, zero_label in zip(gt_probs, zero_labels)])
            gen_probs = [discriminator(pred.detach()).squeeze(1)\
                         for discriminator in self.discriminator]
            gen_loss = sum([self.critic["L2"](gen_prob, one_label)\
                            for gen_prob, one_label in zip(gen_probs, one_labels)])
            disc_loss = (gen_loss + gt_loss)
            disc_loss.backward()

            self.optimizer.step("discriminator")
            reduced_gt_loss = gt_loss.item()
            reduced_gen_loss = gen_loss.item()
            reduced_gt_mean_prob0 = gt_probs[0].mean().item()
            reduced_gen_mean_prob0 = gen_probs[0].mean().item()
            reduced_gt_mean_prob1 = gt_probs[1].mean().item()
            reduced_gen_mean_prob1 = gen_probs[1].mean().item()

            train_losses['train/l1'].append(reduced_l1_loss)
            train_losses['train/gan'].append(reduced_gan_loss)
            train_losses['train/gt'].append(reduced_gt_loss)
            train_losses['train/gen'].append(reduced_gen_loss)
            train_losses['train/gt-prob0'].append(reduced_gt_mean_prob0)
            train_losses['train/gen-prob0'].append(reduced_gen_mean_prob0)
            train_losses['train/gt-prob1'].append(reduced_gt_mean_prob1)
            train_losses['train/gen-prob1'].append(reduced_gen_mean_prob1)

            self.optimizer.scheduler()
        train_losses = {k: np.mean(value) for k, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        return train_losses

    def validation(self, dataloader):
        losses = []
        self.generator.eval()
        val_losses = defaultdict(list)
        for idx, data in enumerate(dataloader):
            lq = data["LQ"].to(self.device)
            gt = data["GT"].to(self.device)
            # generator
            if self.use_pitch:
                pitch = data["Pitch"].to(self.device)
            else:
                pitch = None

            with torch.no_grad():
                pred = self.generator(lq, pitch)
                loss = self.critic["L1"](pred, gt)

            reduced_loss = loss.item()
            losses.append(reduced_loss)
            val_losses['eval/l1'].append(reduced_loss)

        val_losses = {k: np.mean(value) for k, value in val_losses.items()}
        return val_losses


    def save(self, save_dir):
        torch.save(self.generator.state_dict(), save_dir + "generator.pth")
        torch.save([d.state_dict() for d in self.discriminator], save_dir + "discriminator.pth")
        torch.save(self.optimizer.state_dict(), save_dir + "optimizer.pth")

    def load(self, model_dir, optimizer=True, discriminator=True):
        print("load model")
        state_dict = torch.load(model_dir + "generator.pth")
        self._load(state_dict, self.generator)

        if self.discriminator is not None and discriminator:
            [d.load_state_dict(p) for d, p in zip(self.discriminator, torch.load(model_dir + "discriminator.pth"))]

        if self.optimizer is not None and optimizer:
            self.optimizer.load_state_dict(torch.load(model_dir + "optimizer.pth"))

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    print('%s is not in model' % key)
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    print("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                print("not exist ", key)

    def _get_lr(self):
        for param_group in self.optimizer.optimizers['generator'].param_groups:
            lr = param_group['lr']
            break
        return lr

