#coding: utf-8
"""
TODO:
- make TestDataset
- separate transforms
"""

import os
import os.path as osp
import time
import random
import numpy as np
import random
import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 n_fft=2048,
                 hop_length=300,
                 win_length=1200,
                 n_mels=80,
                 max_mel_length=384,
                 use_pitch=False,
                 data_augmentation=False,
                 validation=False,
                 ):
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.sr = sr
        self.mean, self.std = -4, 4
        self.max_mel_length = max_mel_length
        self.use_pitch = use_pitch
        logger.debug('sr: %d\nn_fft: %d\nhop_length: %d\nwin_length: %d' % (sr, n_fft, hop_length, win_length))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        gt, lq, pitch, speaker_id = self._load_tensor(data)
        return gt, lq, pitch, speaker_id

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        path = wave_path.replace('MultiDataset', 'DataForAE')
        data = torch.load(path)
        if len(data) == 2:
            gt, lq = data
            pitch = None
        else:
            gt, lq, pitch = data

        if isinstance(lq, list):
            lq = random.choice(lq)


        mel_size = gt.size(1)
        if mel_size > self.max_mel_length:
            lidx = np.random.randint(0, mel_size - self.max_mel_length)
            gt = gt[:, lidx:lidx+self.max_mel_length]
            lq = lq[:, lidx:lidx+self.max_mel_length]
            if pitch is not None:
                pitch = pitch[lidx:lidx+self.max_mel_length]

        return gt, lq, pitch, int(speaker_id)


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False, adaptive_batch_size=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.adaptive_batch_size = adaptive_batch_size
        self.max_mel_length = 384

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])

        gts = torch.zeros((batch_size, 1, nmels, self.max_mel_length)).float()
        lqs = torch.zeros((batch_size, 1, nmels, self.max_mel_length)).float()
        pitches = torch.zeros((batch_size, 1, self.max_mel_length)).float()
        speaker_ids = torch.zeros((batch_size,)).long()
        for bid, (gt, lq, pitch, spk_id) in enumerate(batch):
            mel_size = gt.size(1)
            gts[bid, 0, :, :mel_size] = gt
            lqs[bid, 0, :, :mel_size] = lq
            if pitch is not None:
                pitches[bid, 0, :mel_size] = pitch
            speaker_ids[bid] = spk_id
        return {"GT": gts, "LQ": lqs, "Pitch": pitches, "Spkid": speaker_ids}


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    dataset = FilePathDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
