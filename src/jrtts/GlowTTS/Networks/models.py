'''
adapted from https://github.com/jaywalnut310/glow-tts/blob/master/models.py
Copyright (c) 2020 Jaehyeon Kim
MIT License
https://opensource.org/licenses/mit-license.php
'''


import math
import torch
from torch import nn
from torch.nn import functional as F

from . import modules, commons, attentions, monotonic_align
from .global_style_tokens import ExtraEmbedding, PitchEmbedding

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = attentions.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = attentions.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
            self,
            n_vocab,
            out_channels,
            hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=None,
            block_length=None,
            mean_only=False,
            prenet=False,
            gin_channels=0,
            use_pitch_embedding=True
    ):

        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.prenet = prenet
        self.gin_channels = gin_channels
        self.use_pitch_embedding = use_pitch_embedding
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if prenet:
            self.pre = modules.ConvReluNorm(hidden_channels, hidden_channels,
                                            hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
        )

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)

        if use_pitch_embedding:
            self.proj_pitch = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp,
                                                kernel_size, p_dropout)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
        x = torch.transpose(x, 1, -1) # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)

        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(x_dp, x_mask)
        if self.use_pitch_embedding:
            x_pitch = self.proj_pitch(x_dp, x_mask)
            x_pitch *= x_mask

        else:
            x_pitch = None
        return x_m, x_logs, logw, x_mask, x_pitch


class FlowSpecDecoder(nn.Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_blocks,
            n_layers,
            p_dropout=0.,
            n_split=4,
            n_sqz=2,
            sigmoid_scale=False,
            gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                attentions.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale))

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()


class FlowGenerator(nn.Module):
    def __init__(self,
                 n_vocab,
                 hidden_channels,
                 filter_channels,
                 filter_channels_dp,
                 out_channels,
                 kernel_size=3,
                 n_heads=2,
                 n_layers_enc=6,
                 p_dropout=0.,
                 n_blocks_dec=12,
                 kernel_size_dec=5,
                 dilation_rate=5,
                 n_block_layers=4,
                 p_dropout_dec=0.,
                 n_speakers=0,
                 n_split=4,
                 n_sqz=1,
                 sigmoid_scale=False,
                 window_size=None,
                 block_length=None,
                 mean_only=False,
                 hidden_channels_enc=None,
                 hidden_channels_dec=None,
                 prenet=False,
                 gin_channels=0,
                 pitch_emb_dim=0,
                 speaker_emb_dim=0,
                 gst_emb_dim=0,
                 n_extra_layers=0,
                 **kwargs):

        super().__init__()

        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.p_dropout = p_dropout
        self.n_blocks_dec = n_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.n_block_layers = n_block_layers
        self.p_dropout_dec = p_dropout_dec
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels or (speaker_emb_dim + gst_emb_dim)
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.prenet = prenet
        self.pitch_emb_dim = pitch_emb_dim

        self.encoder = TextEncoder(
            n_vocab,
            out_channels,
            hidden_channels_enc or hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers_enc,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            mean_only=mean_only,
            prenet=prenet,
            gin_channels=self.gin_channels,
            use_pitch_embedding=(self.pitch_emb_dim > 0)
        )

        self.decoder = FlowSpecDecoder(
            out_channels,
            hidden_channels_dec or hidden_channels,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_dec,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=self.gin_channels + self.pitch_emb_dim)

        self.use_speaker_embedding = (n_speakers > 1)
        if self.use_speaker_embedding:
            # nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)
            self.emb_g = ExtraEmbedding(
                spk_emb_dim=speaker_emb_dim, gst_emb_dim=gst_emb_dim,
                n_speakers=n_speakers, n_extra_layers=n_extra_layers, gin_channels=self.gin_channels)

        self.use_pitch_embedding = (self.pitch_emb_dim > 0)
        if self.use_pitch_embedding:
            self.pitch_emb = PitchEmbedding(self.pitch_emb_dim)

    def forward(self, x, x_lengths, y=None, y_lengths=None,
                g=None, gen=False, noise_scale=1., length_scale=1.,
                pitch=None, pitch_bias=0.):

        ### speaker embedding
        if self.use_speaker_embedding and (g is not None):
            g_enc = self.emb_g(mel=y.transpose(1, 2), spk_id=g.unsqueeze(1))
            g_enc = F.normalize(g_enc)
        else:
            g_enc = None

        ### text encoding
        x_m, x_logs, logw, x_mask, x_pitch = self.encoder(x, x_lengths, g=g_enc)

        ### duration postprocess
        if gen:
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)
        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        ### decoding
        if gen:
            attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            g_dec = self._get_g_dec(x_pitch, pitch, pitch_bias, g_enc, attn)

            y_m = torch.matmul(attn.squeeze(1).transpose(1, 2),
                               x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            y_logs = torch.matmul(attn.squeeze(1).transpose(1, 2),
                                  x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
            z = (y_m + torch.exp(y_logs) * torch.randn_like(y_m) * noise_scale) * y_mask
            y, logdet = self.decoder(z, y_mask, g=g_dec, reverse=True)
            return (y, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs, pitch
        else:
            g_dec = self._get_g_dec(x_pitch=None,
                                    pitch=pitch,
                                    pitch_bias=0,
                                    g_enc=g_enc)
            z, logdet = self.decoder(y, y_mask, g=g_dec, reverse=False)
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
                logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
                logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
                logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
                logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

            attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
            if self.use_pitch_embedding:
                mean_pitch = self.pitch_emb.meaned_pitch(pitch, attn).transpose(1, 2)
                mean_pitch *= x_mask
            else:
                mean_pitch = None
            y_m = torch.matmul(attn.squeeze(1).transpose(1, 2),
                               x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            y_logs = torch.matmul(attn.squeeze(1).transpose(1, 2),
                                  x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
            return (z, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs, mean_pitch, x_pitch


    def _get_g_dec(self, x_pitch, pitch=None, pitch_bias=0, g_enc=None, attn=None):
        if not (self.use_pitch_embedding or self.use_speaker_embedding):
            return None

        if self.use_pitch_embedding:
            if attn is not None:
                x_pitch += pitch_bias
                pitch = self.pitch_emb.pitch_expansion(x_pitch.squeeze(1), attn)
            else:
                assert(pitch is not None)
            pitch_emb = self.pitch_emb(pitch)
        else:
            pitch_emb = torch.zeros((g_enc.size(0), 0, 1))

        if self.use_speaker_embedding:
            g_dec = torch.cat([g_enc.expand(-1, -1, pitch_emb.size(-1)), pitch_emb], dim=1)
        else:
            g_dec = pitch_emb

        return g_dec



    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:,:,:y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()


    def forward_with_target_alignment(self,
                                      x, x_lengths, y=None, y_lengths=None,
                                      g=None, noise_scale=1., length_scale=1.,
                                      pitch_bias=0.,
                                      pitch=None):

        if self.use_speaker_embedding and (g is not None):
            g_enc = self.emb_g(mel=y.transpose(1, 2), spk_id=g.unsqueeze(1))
            g_enc = F.normalize(g_enc)
        else:
            g_enc = None

        x_m, x_logs, logw, x_mask, x_pitch = self.encoder(x, x_lengths, g=g_enc)
        y_max_length = y.size(2)
        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        g_dec = self._get_g_dec(x_pitch, pitch, pitch_bias, g_enc)
        z, logdet = self.decoder(y, y_mask, g=g_dec, reverse=False)
        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']
            attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        if pitch is not None:
            mean_pitch = self.pitch_emb.meaned_pitch(pitch, attn).transpose(1, 2)
            mean_pitch *= x_mask
            # gen
            pitch = self.pitch_emb.pitch_expansion(x_pitch.squeeze(1), attn)
        g_dec = self._get_g_dec(x_pitch, pitch, pitch_bias, g_enc)

        y_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        y_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
        z = (y_m + torch.exp(y_logs) * torch.randn_like(y_m) * noise_scale) * y_mask
        y, logdet = self.decoder(z, y_mask, g=g_dec, reverse=True)

        return (y, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs, pitch


    def controled_generation(self, x, y, attn=None, pitch=None, noise_scale=1., length_scale=1.):
        pass

    def pitch_preprocess(self, f0, center_index=3):
        """
        f0 (str):
        """
        unit = (torch.log(torch.FloatTensor([2])) / 12) / 5

        if len(f0) == 0:
            f0_tensor = unit * (int(f0) - center_index)
        else:
            index_tensor = torch.LongTensor(list(map(int, list(f0))))
            f0_tensor = (index_tensor - center_index) * unit
        f0_tensor = f0_tensor.reshape(1, 1, -1)
        return f0_tensor
