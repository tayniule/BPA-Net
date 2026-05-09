import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
import numpy as np



class SKAttention2D(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=(k, k), padding=(k // 2, k // 2), groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for _ in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, h, w = x.size()
        conv_outs = []

        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = U.mean(dim=[-2, -1])
        Z = self.fc(S)

        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))

        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)

        return V


class EEGNet(nn.Module):
    def __init__(self, batch_size=128, seq_len=800, n_channels=18, n_classes=2):
        super(EEGNet, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.block2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1 * D, kernel_size=(1, 48), padding=(0, 8), bias=False),
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.2)
        )

        # 全局池化确保固定输出维度
        self.merge_s2 = nn.Sequential(
            nn.AvgPool2d(1),
            SKAttention2D(F1 * D, reduction=4),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        print_shape("EEGNet input", x) #(B, 18, 800)
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        print_shape("EEGNet reshaped", x) #(B, 1, 18, 800)
        x = self.get_feature(x)
        print_shape("EEGNet features", x) #(B, 1, F1 * D, 800)
        x = self.merge_s2(x)
        print_shape("EEGNet after pool", x) #(B, 16, 1, 1)
        x = x.view(x.size(0), -1)
        print_shape("EEGNet output", x) #(B, 16)
        return x

class E_COM_GNet(nn.Module):
    def __init__(self, batch_size=128, seq_len=384, n_channels=2, n_classes=2):
        super(E_COM_GNet, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 128),
                padding=(0, 64),
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(1, 48),
                      padding=(0, 8),
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  # Pointwise
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.2)
        )
        self.merge_s2 = nn.Sequential(
            nn.AvgPool2d(1 * 1),
            SKAttention2D(F1 * D, reduction=4)
            # Permute([0, 2, 1])
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        # x = x.reshape(64, 16, -1)
        # print(x.size())
        x = self.merge_s2(x)
        x = x.reshape(-1, x.shape[-1])
        # print('E_COM_GNet:', x.size())

        return x

class GSRNet(nn.Module):
    def __init__(self, batch_size=128, seq_len=800, n_channels=1, n_classes=2):
        super(GSRNet, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1 * D, kernel_size=(1, 1), groups=1, bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 48), padding=(0, 8), groups=F1 * D,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.2)
        )

        self.merge_s2 = nn.Sequential(
            nn.AvgPool2d(1),
            SKAttention2D(F1 * D, reduction=4),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        print_shape("GSRNet input", x)
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        print_shape("GSRNet reshaped", x)
        x = self.get_feature(x)
        print_shape("GSRNet features", x)
        x = self.merge_s2(x)
        print_shape("GSRNet after pool", x)
        x = x.view(x.size(0), -1)
        print_shape("GSRNet output", x)
        return x


class PPGFeatureExtractor(nn.Module):
    def __init__(self, batch_size=128, seq_len=800, n_channels=1, n_classes=2):
        super(PPGFeatureExtractor, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1 * D, kernel_size=(1, 1), groups=1, bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 48), padding=(0, 8), groups=F1 * D,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.2)
        )

        self.merge_s2 = nn.Sequential(
            nn.AvgPool2d(1),
            SKAttention2D(F1 * D, reduction=4),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        print_shape("PPGNet input", x)
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        print_shape("PPGNet reshaped", x)
        x = self.get_feature(x)
        print_shape("PPGNet features", x)
        x = self.merge_s2(x)
        print_shape("PPGNet after pool", x)
        x = x.view(x.size(0), -1)
        print_shape("PPGNet output", x)
        return x
