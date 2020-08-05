#coding: utf-8
"""
TODO:
- make TestDataset
- separate transforms
"""

import os
import os.path as osp
import random
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)

np.random.seed(1)
random.seed(1)

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, augmentation=False, sr=24000):
        self.data_list = data_list
        self.sr = sr
        self.augmentation = augmentation
        self.aug_p = 0.5
        self.mean, self.std = -4, 4

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wave_tensor, mel_tensor = self._load_tensor(idx)
        return wave_tensor, mel_tensor

    def _load_tensor(self, idx):
        data = self.data_list[idx]
        load_data = torch.load(data)
        if isinstance(load_data, torch.Tensor):
            wave = load_data[0].numpy()
            mel = torch.load(data.replace('wav.pth', 'mel.pth'))[0] # already log scaled
            mel = self._preprocessing(mel)
            mel = mel.transpose(0, 1).numpy()
        elif isinstance(load_data, list):
            wave, _, mel = load_data
            wave = wave.numpy() * 0.99
            mel = mel.numpy()

        return wave, mel

    def _preprocessing(self, mel):
        normed_mel = (mel - self.mean) / self.std
        if self.augmentation:
            normed_mel = self._augmentation(normed_mel, p=self.aug_p)
        return normed_mel

    def _augmentation(self, mel, p=0.5):
        if np.random.random() < p:
            mel = smoothing(mel)
        return mel

@torch.no_grad()
def smoothing(x):
    size, pad = random.choices([[3, 1], [(3, 1), (0, 0, 1, 1)], [(1, 5), (2, 2, 0, 0)]],
                               k=1, weights=[0.5, 0.25, 0.25])[0]
    x = x.unsqueeze(0).unsqueeze(1)
    x = torch.nn.ReflectionPad2d(pad)(x)
    x = torch.nn.AvgPool2d(size, stride=1)(x)
    x = x.squeeze(1).squeeze(0)
    return x

class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self,
                 batch_max_steps=24000, # 20480
                 hop_size=300,
                 aux_context_window=0, #2
                 use_noise_input=True,
                 ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.
             - batch[idx][0]: audio (T, )
             - batch[idx][1]: feature (C, T')
        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where T = (T' - 2 * aux_context_window) * hop_size
            Tensor: Target signal batch (B, 1, T).

        """
        # time resolution check
        y_batch, c_batch = [], []
        for idx in range(len(batch)):
            x, c = batch[idx]
            x, c = self._adjust_length(x, c)
            self._check_length(x, c, self.hop_size, 0)
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                interval_start = self.aux_context_window
                interval_end = len(c) - self.batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step: start_step + self.batch_max_steps]
                c = c[start_frame - self.aux_context_window:
                      start_frame + self.aux_context_window + self.batch_max_frames]
                self._check_length(y, c, self.hop_size, self.aux_context_window)
            else:
                logger.warn(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.astype(np.float32).reshape(-1, 1)]  # [(T, 1), (T, 1), ...]
            c_batch += [c.astype(np.float32)]  # [(T' C), (T' C), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        c_batch = torch.FloatTensor(np.array(c_batch)).transpose(2, 1)  # (B, C, T')

        # make input noise signal batch tensor
        if self.use_noise_input:
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            return (z_batch, c_batch), y_batch
        else:
            return (c_batch,), y_batch

    def _adjust_length(self, x, c):
        """Adjust the audio and feature lengths.

        NOTE that basically we assume that the length of x and c are adjusted
        in preprocessing stage, but if we use ESPnet processed features, this process
        will be needed because the length of x is not adjusted.

        """
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")
        return x, c

    @staticmethod
    def _check_length(x, c, hop_size, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * hop_size, (len(x), len(c), (len(c) - 2 * context_window)*hop_size)


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     collate_config={}):
    dataset = FilePathDataset(path_list, augmentation=(not validation))
    collate_fn = Collater(**collate_config)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=True)

    return data_loader
