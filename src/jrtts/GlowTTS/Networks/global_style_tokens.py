'''
adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
Copyright (c) 2018 MagicGirl Sakura
MIT License
https://opensource.org/licenses/mit-license.php
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
np.random.seed(0)

class PitchEmbedding(nn.Module):
    def __init__(self, pitch_emb_dim=32):
        super(PitchEmbedding, self).__init__()
        self.pitch_embedding = nn.Conv1d(1, pitch_emb_dim, kernel_size=5, padding=2)
        self.avg_pool = nn.AvgPool1d(kernel_size=6, stride=2, padding=2)
        #self.avg_pool = nn.AvgPool1d(2)
        self.initialize()

    def forward(self, pitch):
        pitch_emb = self.pitch_embedding(pitch.unsqueeze(1))
        pitch_emb = self.avg_pool(pitch_emb)
        return pitch_emb

    def meaned_pitch(self, pitch, attn):
        _, mean_pitch = self.pitch_regulation(pitch, attn)
        return mean_pitch

    @torch.no_grad()
    def pitch_regulation(self, pitch, attn):
        attn = attn.float().squeeze(1)
        pitch_nonzero_mask = (pitch > 0).float().unsqueeze(2)
        nonzero_length = torch.bmm(attn, pitch_nonzero_mask).clamp(min=1)
        mean_pitch = torch.bmm(attn, pitch.unsqueeze(2))/ nonzero_length
        reg_pitch = torch.bmm(attn.transpose(1, 2), mean_pitch).squeeze(2)
        return reg_pitch, mean_pitch

    @torch.no_grad()
    def pitch_expansion(self, mean_pitch, attn):
        attn = attn.float().squeeze(1)
        pitch = torch.bmm(attn.transpose(1, 2), mean_pitch.unsqueeze(2)).squeeze(2)
        return pitch

    def initialize(self):
        for param in self.pitch_embedding.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
            else:
                nn.init.uniform_(param, -0.001, 0.001)

class ExtraEmbedding(nn.Module):
    def __init__(self,
                 n_speakers=120,
                 spk_emb_dim=128,
                 gst_emb_dim=128,
                 nmels=80,
                 n_extra_layers=0,
                 gin_channels=None,
                 random_mel_slice=True,
                 ):
        super(ExtraEmbedding, self).__init__()
        self.nmels = nmels
        self.n_speakers = n_speakers
        self.spk_emb_dim = spk_emb_dim
        self.gst_emb_dim = gst_emb_dim
        self.gin_channels = gin_channels or (spk_emb_dim + gst_emb_dim)
        self.gst = GST(stl_token_embedding_size=gst_emb_dim) #gst_input.shape must be (b, mlen, nmels)
        self.spk_embedding = nn.Embedding(n_speakers, spk_emb_dim)
        self.random_mel_slice = random_mel_slice
        self.n_extra_layers = n_extra_layers
        self.dropout_p = 0.25
        if self.n_extra_layers > 0:
            self.extra_layers = self._get_extra_layers()
        self.initialize()

    def forward(self, mel, spk_id):
        assert(mel.shape[-1] == self.nmels)
        assert(len(spk_id.shape) == 2)
        if (self.training and self.random_mel_slice):
            random_slice_length = mel.shape[1] // 4
            l_rand = np.random.randint(0, random_slice_length // 2)
            r_rand = np.random.randint(1, random_slice_length // 2)
            mel = mel[:, l_rand:-r_rand]

        style_emb = self.gst(mel)
        spk_emb = self.spk_embedding(spk_id)
        g = torch.cat([spk_emb, style_emb], dim=2)
        if self.n_extra_layers > 0:
            g = self.extra_layers(g)
        g = g.transpose(1, 2)
        return g

    def initialize(self):
        nn.init.uniform_(self.spk_embedding.weight, -0.1, 0.1)
        for param in self.gst.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
            else:
                nn.init.uniform_(param, -0.001, 0.001)

    def _get_extra_layers(self):
        channels = [self.spk_emb_dim + self.gst_emb_dim] + [self.gin_channels] * self.n_extra_layers
        extra_layers = [nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(), nn.Dropout(self.dropout_p)) for in_c, out_c in zip(channels[:-1], channels[1:])]
        extra_layers += [nn.Linear(channels[-1], self.gin_channels)]
        extra_layers = nn.Sequential(*extra_layers)
        return extra_layers

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(
            self,
            ref_enc_filters=[32, 32, 64, 64, 128, 128],
            ref_enc_size=[3, 3],
            ref_enc_strides=[2, 2],
            ref_enc_pad=[1, 1],
            ref_enc_gru_size=64,
            n_mel_channels=80
    ):

        super().__init__()
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = n_mel_channels
        self.ref_enc_gru_size = ref_enc_gru_size

    def forward(self, inputs):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self,
                 token_embedding_size=128,
                 token_num=10,
                 num_heads=4):
        super().__init__()
        self.embed = nn.Parameter(
            torch.FloatTensor(token_num, token_embedding_size // num_heads))
        d_q = token_embedding_size // 2
        d_k = token_embedding_size // num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=token_embedding_size,
            num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self,
                 stl_token_embedding_size=128,
                 stl_token_num=10,
                 stl_num_heads=4
    ):
        super().__init__()
        self.encoder = ReferenceEncoder(ref_enc_gru_size=stl_token_embedding_size//2)
        self.stl = STL(
            token_embedding_size=stl_token_embedding_size,
            token_num=stl_token_num,
            num_heads=stl_num_heads)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed
