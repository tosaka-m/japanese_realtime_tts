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

from .text_utils import TextCleaner
np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt') #'kata_dict.csv') #

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 dict_path=DEFAULT_DICT_PATH,
                 sr=24000,
                 n_fft=2048,
                 hop_length=300,
                 win_length=1200,
                 n_mels=80,
                 data_augmentation=False,
                 validation=False,
                 ):
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner(dict_path)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.sr = sr
        self.mean, self.std = -4, 4
        self.std_f0 = 5
        logger.debug('sr: %d\nn_fft: %d\nhop_length: %d\nwin_length: %d' % (sr, n_fft, hop_length, win_length))

        self.data_augmentation = data_augmentation and (not validation)
        if data_augmentation:
            self.extend_data_list = [d for d in self.data_list if len(d[1]) < 60]
            self.mel_zero_pad = (torch.log(1e-4 + torch.zeros((n_mels, int(sr*0.6 // hop_length))).float()) - self.mean) / self.std
            self.wave_zero_pad = torch.zeros((int(sr*0.6), )).float()
            self.f0_zero_pad = torch.zeros((self.mel_zero_pad.shape[1], )).float()
            self.text_zero_pad = torch.LongTensor(self.text_cleaner(' '))
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        wave_tensor, mel_tensor, text_tensor, f0_tensor, speaker_id = self._load_tensor(data)
        if self.data_augmentation and mel_tensor.shape[1] < 600 and np.random.random() < 0.8:
            ext_data = random.choice(self.extend_data_list)
            ext_wave_tensor, ext_mel_tensor, ext_text_tensor, ext_f0_tensor, ext_speaker_id\
                = self._load_tensor(ext_data)
            wave_tensor = self._concat_data(wave_tensor, ext_wave_tensor, data_type='wave')
            mel_tensor = self._concat_data(mel_tensor, ext_mel_tensor, data_type='mel')
            text_tensor = self._concat_data(text_tensor, ext_text_tensor, data_type='text')
            f0_tensor = self._concat_data(f0_tensor, ext_f0_tensor, data_type='f0')

        return wave_tensor, mel_tensor, text_tensor, f0_tensor, speaker_id, data[0]

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        waves = torch.load(wave_path, map_location='cpu')
        wave, mel, f0 = waves

        text = torch.LongTensor(self.text_cleaner(text))
        mel = (torch.log(1e-5 + mel) - self.mean) / self.std
        mel_len = mel.size(1) - mel.size(1) % 2
        mel = mel[:, :mel_len]
        f0 = f0[:mel_len]
        f0 = torch.log(1 + f0) / self.std_f0
        return wave, mel, text, f0, speaker_id

    def _concat_data(self, data1, data2, data_type='wave'):

        if data_type == 'wave':
            cat_data = torch.cat([data1, self.wave_zero_pad, data2], dim=0)
        elif data_type == 'mel':
            cat_data = torch.cat([data1, self.mel_zero_pad, data2], dim=1)
        elif data_type == 'text':
            cat_data = torch.cat([data1, self.text_zero_pad, data2], dim=0)
        elif data_type == 'f0':
            cat_data = torch.cat([data1, self.f0_zero_pad, data2], dim=0)
        return cat_data


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False, adaptive_batch_size=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.adaptive_batch_size = adaptive_batch_size
        self.max_mel_volume = 16 * 80 * 1000

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        f0s = torch.zeros((batch_size, max_mel_length)).float()
        speaker_ids = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        for bid, (_, mel, text, f0, speaker_id, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            speaker_ids[bid] = speaker_id
            f0s[bid, :mel_size] = f0
            paths[bid] = path

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, f0s, speaker_ids, paths, waves

        return texts, input_lengths, mels, output_lengths, f0s, speaker_ids, paths

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
