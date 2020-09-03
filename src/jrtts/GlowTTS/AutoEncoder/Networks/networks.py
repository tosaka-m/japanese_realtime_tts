import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, hidden_dim=64, depth=4, pitch=False):
        super(UNet, self).__init__()
        if pitch:
            pitch_emb_dim = 16
            self.pitch_emb = Conv1dBlock(in_c=1, out_c=pitch_emb_dim)
        else:
            pitch_emb_dim = 0

        down_in_channels  = [hidden_dim + pitch_emb_dim] + [hidden_dim * (2**i) for i in range(1, 1+depth)]
        down_out_channels = [hidden_dim * (2**i) for i in range(1, depth)] + [hidden_dim * (2**(depth-1))]
        up_in_channels    = [hidden_dim * (2**(i-1)) for i in range(depth, 0, -1)]
        up_out_channels   = [hidden_dim * (2**(i-1))  for i in range(depth-1, 0, -1)] + [hidden_dim]
        cat_channels = [hidden_dim * (2**(i-1)) for i in range(depth, 1, -1)] + [hidden_dim + pitch_emb_dim]
        self.init_conv = Conv2dBlock(in_c, hidden_dim)
        self.last_conv = nn.Conv2d(hidden_dim, out_c, kernel_size=1)
        self.down_convs = nn.ModuleList([DownSample2d(down_in_c, down_out_c) \
                           for down_in_c, down_out_c in zip(down_in_channels, down_out_channels)])
        self.up_convs = nn.ModuleList([UpSample2d(up_in_c, up_out_c, cat_c=cat_c) \
                         for up_in_c, up_out_c, cat_c in zip(up_in_channels, up_out_channels, cat_channels)])
        self.use_pitch = pitch

    def forward(self, x, pitch=None):
        x = self.init_conv(x)
        if self.use_pitch:
            p = self.pitch_emb(pitch)
            x = torch.cat([x, p.unsqueeze(2).expand(-1, -1, x.size(2), -1)], dim=1)

        x_downs = []
        for down_conv in self.down_convs:
            x_downs.append(x)
            x = down_conv(x)
        for x_down, up_conv in zip(x_downs[::-1], self.up_convs):
            x = up_conv(x, x_down)

        x = self.last_conv(x)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super(Conv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Conv1dBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super(Conv1dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class DownSample2d(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSample2d, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.block = Conv2dBlock(in_c, out_c)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.block(x)
        return x

class UpSample2d(nn.Module):
    def __init__(self, in_c, out_c, cat_c=None):
        super(UpSample2d, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        cat_c = cat_c if cat_c is not None else in_c
        self.block = Conv2dBlock(in_c + cat_c, out_c)

    def forward(self, x, y):
        x = self.upsample(x)
        zero_pad = nn.ZeroPad2d((0, y.size(3)-x.size(3), 0, y.size(2)-x.size(2)))
        x = zero_pad(x)
        x = torch.cat([y, x], dim=1)
        x = self.block(x)
        return x

class OrigDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm='sn'):
        super(OrigDiscriminator, self).__init__()
        if norm == 'sn':
            conv_wrapper = lambda x: nn.utils.spectral_norm(x)
        else:
            conv_wrapper = lambda x: x

        model = [nn.ReflectionPad2d((2,2,1,1)),
                 conv_wrapper(nn.Conv2d(input_nc, ndf, kernel_size=(4, 5), stride=2,
                                        padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      conv_wrapper(nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=(3, 4), stride=(1, 2),
                                             padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  conv_wrapper(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=3, stride=2,
                                padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        self.conv = conv_wrapper(
            nn.Conv2d(ndf*mult*2, 1,
                      kernel_size=3, stride=1, padding=0, bias=False))
        self.model = nn.ModuleList(model)
        self.phase_shuffle = PhaseShuffle2d(n=1)


    def forward(self, x):
        for idx, model in enumerate(self.model):
            x = model(x)
            if isinstance(model, nn.ReflectionPad2d) and idx < 8:
                x = self.phase_shuffle(x)
        x = self.conv(x).squeeze(1)

        return x

    def feature_extract(self, x):
        for model in self.model[:3]:
            x = model(x)
        return x

class PhaseShuffle2d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle2d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :, :, :move]
            right = x[:, :, :, move:]
            shuffled = torch.cat([right, left], dim=3)
        return shuffled

class PhaseShuffle1d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle1d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :,  :move]
            right = x[:, :, move:]
            shuffled = torch.cat([right, left], dim=2)

        return shuffled

class OrigDiscriminator1d(nn.Module):
    def __init__(self, input_nc=80, ndf=128, n_layers=5, norm='sn'):
        super(OrigDiscriminator1d, self).__init__()
        if norm == 'sn':
            conv_wrapper = lambda x: nn.utils.spectral_norm(x)
        else:
            conv_wrapper = lambda x: x

        model = [nn.ReflectionPad1d(1),
                 conv_wrapper(
                     nn.Conv1d(input_nc, ndf, kernel_size=5, stride=2,
                               padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad1d(1),
                      conv_wrapper(
                          nn.Conv1d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2,
                                    padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad1d(1),
                  conv_wrapper(
                      nn.Conv1d(ndf * mult, ndf * mult, kernel_size=4, stride=2,
                            padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        self.conv = conv_wrapper(
            nn.Conv1d(ndf * mult, 1, kernel_size=3, stride=1, padding=0, bias=False))

        self.model = nn.ModuleList(model)
        self.phase_shuffle = PhaseShuffle1d(n=1)

    def forward(self, x):
        x = x.squeeze(1)
        for idx, model in enumerate(self.model):
            x = model(x)
            if isinstance(model, nn.ReflectionPad1d) and idx < 10:
                x = self.phase_shuffle(x)
        x = self.conv(x).squeeze(1)
        return x

    def feature_extract(self, x):
        x = x.squeeze(1)
        for model in self.model[:6]:
            x = model(x)
        return x


