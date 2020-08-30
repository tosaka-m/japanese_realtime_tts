#coding:utf-8
import copy
import torch
from torch import nn


def build_critic():
    #pl = PerceptualLoss()
    critic = {
        "L1": torch.nn.L1Loss(),
        "L2": torch.nn.MSELoss(),
        "MultiL1": MultiLoss(torch.nn.L1Loss()),
        "MultiL1": MultiLoss(torch.nn.MSELoss()),
        "CE": torch.nn.CrossEntropyLoss(),
        #"PL": lambda x, y: pl.get_loss(x, y),
        }
    return critic



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        import torchvision
        self.vgg = torchvision.models.vgg11(pretrained=True).eval().cuda()
        self.mse = nn.MSELoss().cuda()
        self.max_depth = 17
        self.mean, self.std = -4.2, 2.0

    def forward(self, x):
        outputs = []
        for model in self.vgg.features[:self.max_depth]:
            x = model(x)
            if isinstance(model, torch.nn.Conv2d):
                outputs.append(x)

        return outputs

    def get_loss(self, x, target):
        #x = (torch.stack([x]*3, dim=1) - self.mean) / self.std
        #y = (torch.stack([target]*3, dim=1) - self.mean) / self.std
        x = torch.cat([x]*3, dim=1)
        y = torch.cat([target]*3, dim=1)
        x_outs = self.forward(x)
        y_outs = self.forward(y)
        loss = sum([self.mse(ox, oy.detach()) for ox, oy in zip(x_outs, y_outs)])
        return loss


class MultiLoss(nn.Module):
    def __init__(self, loss_fn):
        super(MultiLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, x, target):
        loss = 0
        for _x, _t in zip(x, target):
            loss += self.loss_fn(_x, _t)
        return loss

