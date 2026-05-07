import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
import numpy as np

# 全局调试标志 - 第一次运行时设为True查看维度
DEBUG = False


def print_shape(name, tensor):
    """打印张量形状"""
    if DEBUG and tensor is not None:
        print(f"[SHAPE] {name:30s}: {str(tuple(tensor.shape)):20s} | "
              f"total={tensor.numel():6d} | mean={tensor.mean().item():.3f} | "
              f"std={tensor.std().item():.3f}")


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


class MDNet(nn.Module):
    def __init__(self, config, params):
        super(MDNet, self).__init__()
        self.config = config
        self.params = params

        print(f"\n{'=' * 60}")
        print(f"Initializing MDNet for dataset: {config.dataset_name}")
        print(f"{'=' * 60}")

        # 根据数据集配置
        if config.dataset_name == 'DEAP':
            eeg_channels, mod2_channels, mod3_channels = 32, 2, 2
            eeg_shape = mod2_shape = mod3_shape = 16  # 全局池化后
            self.eeg_conv_input_size = 528  # 经过eeg_conv后的实际维度
            self.e_com_g_conv_input_size = 528

        elif config.dataset_name == 'HCI':
            eeg_channels, mod2_channels, mod3_channels = 32, 3, 1
            eeg_shape = mod2_shape = mod3_shape = 16
            self.eeg_conv_input_size = 528
            self.e_com_g_conv_input_size = 528

        elif config.dataset_name == 'MER':
            # MER: 18 EEG + 1 GSR + 1 PPG
            eeg_channels, mod2_channels, mod3_channels = 18, 1, 1
            eeg_shape = mod2_shape = mod3_shape = 16  # 全局池化输出16维

            # 关键：这些值需要根据实际调试输出确定
            # 先设为None，在第一次forward时动态计算
            self.eeg_conv_input_size = None  # 将在第一次forward时计算
            self.e_com_g_conv_input_size = None

        else:
            raise ValueError(f'Invalid dataset name: {config.dataset_name}')

        self.eeg_shape = eeg_shape
        self.mod2_shape = mod2_shape
        self.mod3_shape = mod3_shape

        print(f"Feature extractors: EEG({eeg_channels}ch), MOD2({mod2_channels}ch), MOD3({mod3_channels}ch)")
        print(f"Shape config: eeg={eeg_shape}, mod2={mod2_shape}, mod3={mod3_shape}")

        # 特征提取器
        self.EEGFeatureExtractor = EEGNet(
            batch_size=config.batch_size,
            seq_len=800 if config.dataset_name == 'MER' else 512,
            n_channels=eeg_channels,
            n_classes=config.num_classes
        )

        if config.dataset_name == 'MER':
            self.GSRFeatureExtractor = GSRNet(
                batch_size=config.batch_size,
                seq_len=800,
                n_channels=mod2_channels,
                n_classes=config.num_classes
            )
            self.PPGFeatureExtractor = PPGFeatureExtractor(
                batch_size=config.batch_size,
                seq_len=800,
                n_channels=mod3_channels,
                n_classes=config.num_classes
            )
        else:
            self.EOGFeatureExtractor = E_COM_GNet(batch_size=config.batch_size, seq_len=512, n_channels=mod2_channels,
                                                  n_classes=config.num_classes)
            self.EMGFeatureExtractor = E_COM_GNet(batch_size=config.batch_size, seq_len=512, n_channels=mod3_channels,
                                                  n_classes=config.num_classes)
            self.ECGFeatureExtractor = E_COM_GNet(batch_size=config.batch_size, seq_len=512, n_channels=mod2_channels,
                                                  n_classes=config.num_classes)
            self.GSRFeatureExtractor = GSRNet(batch_size=config.batch_size, seq_len=512, n_channels=mod3_channels,
                                              n_classes=config.num_classes)

        self.eeg_input_size = eeg_channels
        self.eog_input_size = mod2_channels
        self.emg_input_size = mod3_channels

        self.eeg_hidden_size = config.hidden_size
        self.eog_hidden_size = config.hidden_size
        self.emg_hidden_size = config.hidden_size
        self.subject_num = subject_num = config.subject_num
        self.hidden_size = hidden_size = config.hidden_size

        self.input_sizes = [self.eeg_input_size, self.eog_input_size, self.emg_input_size]
        self.hidden_sizes = [int(self.eeg_hidden_size), int(self.eog_hidden_size), int(self.emg_hidden_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = config.dropout
        self.activation = config.activation()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.activation2 = nn.ELU()

        # mapping modalities to same sized space
        self.project_eeg = nn.Sequential()
        self.project_eeg.add_module('project_eeg', nn.Linear(in_features=eeg_shape, out_features=config.hidden_size))
        self.project_eeg.add_module('project_eeg_activation', self.activation)
        self.project_eeg.add_module('project_eeg_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_eog = nn.Sequential()
        self.project_eog.add_module('project_eog', nn.Linear(in_features=mod2_shape, out_features=config.hidden_size))
        self.project_eog.add_module('project_eog_activation', self.activation)
        self.project_eog.add_module('project_eog_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_emg = nn.Sequential()
        self.project_emg.add_module('project_emg', nn.Linear(in_features=mod3_shape, out_features=config.hidden_size))
        self.project_emg.add_module('project_emg_activation', self.activation)
        self.project_emg.add_module('project_emg_layer_norm', nn.LayerNorm(config.hidden_size))

        # private encoders
        self.private_eeg = nn.Sequential()
        self.private_eeg.add_module('private_eeg_1',
                                    nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_eeg.add_module('private_eeg_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.private_eeg.add_module('private_eeg_activation_1', self.activation2)

        self.private_eog = nn.Sequential()
        self.private_eog.add_module('private_eog_1',
                                    nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_eog.add_module('private_eog_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.private_eog.add_module('private_eog_activation_1', self.activation2)

        self.private_emg = nn.Sequential()
        self.private_emg.add_module('private_emg_1',
                                    nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_emg.add_module('private_emg_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.private_emg.add_module('private_emg_activation_1', self.activation2)

        # shared encoder
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.shared.add_module('shared_activation_1', self.activation2)

        # fc layers for homogeneous graph distillation
        self.proj1_eeg_low = nn.Linear(in_features=config.hidden_size, out_features=50)
        self.proj2_eeg_low = nn.Linear(in_features=50, out_features=config.hidden_size)
        self.out_layer_eeg_low = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)
        self.proj1_mod2_low = nn.Linear(in_features=config.hidden_size, out_features=50)
        self.proj2_mod2_low = nn.Linear(in_features=50, out_features=config.hidden_size)
        self.out_layer_mod2_low = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)
        self.proj1_mod3_low = nn.Linear(in_features=config.hidden_size, out_features=50)
        self.proj2_mod3_low = nn.Linear(in_features=50, out_features=config.hidden_size)
        self.out_layer_mod3_low = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)

        # fc layers for Heterogeneous graph distillation
        self.proj1_eeg_high = nn.Linear(in_features=config.hidden_size, out_features=100)
        self.proj2_eeg_high = nn.Linear(in_features=100, out_features=config.hidden_size)
        self.out_layer_eeg_high = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)
        self.proj1_mod2_high = nn.Linear(in_features=config.hidden_size, out_features=100)
        self.proj2_mod2_high = nn.Linear(in_features=100, out_features=config.hidden_size)
        self.out_layer_mod2_high = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)
        self.proj1_mod3_high = nn.Linear(in_features=config.hidden_size, out_features=100)
        self.proj2_mod3_high = nn.Linear(in_features=100, out_features=config.hidden_size)
        self.out_layer_mod3_high = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)

        # conv layer
        self.eeg_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 48), padding=(0, 8), bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(17, 1), groups=2, bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(16, 1), groups=2, bias=False),
            nn.ELU()
        )

        self.e_com_g_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 48), padding=(0, 8), bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), groups=2, bias=False),
            nn.ELU()
        )

        # 对于MER，先创建占位，第一次forward时重新创建
        if config.dataset_name == 'MER':
            self.leaner = nn.Linear(1, 64)  # 占位，将被替换
            self.leaner_1 = nn.Linear(1, 64)
            self.leaner_2 = nn.Linear(1, 64)
            self._leaner_initialized = False
        else:
            self.leaner = nn.Linear(in_features=528, out_features=64)
            self.leaner_1 = nn.Linear(in_features=1584, out_features=64)
            self.leaner_2 = nn.Linear(in_features=528, out_features=64)
            self._leaner_initialized = True

        self.bn1 = nn.BatchNorm2d(16)
        self.layer_norm = nn.LayerNorm(64)
        self.layer_norm_1 = nn.LayerNorm(384)

        # subject private encoders
        self.subject_private = []
        if torch.cuda.is_available():
            for i in range(1, self.params["group_num"] + 1):
                self.subject_private.append(nn.Sequential(
                    nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6),
                    self.activation2
                ).cuda())
        else:
            for i in range(1, self.params["group_num"] + 1):
                self.subject_private.append(nn.Sequential(
                    nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6),
                    self.activation2
                ))

        # subject shared encoder
        self.subject_shared = nn.Sequential()
        self.subject_shared.add_module('shared_1', nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6))
        self.subject_shared.add_module('shared_activation_1', self.activation2)
        self.subject_shared.add_module('shared_batch_norm_2', nn.BatchNorm1d(hidden_size * 6))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_batch_norm', nn.BatchNorm1d(self.hidden_size * 6))
        self.fusion.add_module('fusion_layer_1',
                               nn.Linear(in_features=self.hidden_size * 6, out_features=int(self.hidden_size / 4)))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', nn.ELU())
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=int(self.hidden_size / 4), out_features=output_size))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size,
                                                        dim_feedforward=hidden_size * 2, nhead=2,
                                                        dropout=self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.utt_feature_norm = nn.BatchNorm1d(6 * self.hidden_size)

    def extract_features(self, extractor, x):
        features = extractor(x)
        return features

    def alignment(self, eeg, mod2, mod3, lengths, subject_labels, groups):
        batch_size = lengths.size(0)
        # print(f"\n{'=' * 60}")
        # print(f"ALIGNMENT - batch_size={batch_size}")
        # print(f"{'=' * 60}")

        # extract features from eeg modality
        print_shape("Input eeg", eeg)
        utterance_eeg = self.extract_features(self.EEGFeatureExtractor, eeg)
        print_shape("Extracted utterance_eeg", utterance_eeg)

        if self.config.dataset_name == 'DEAP':
            utterance_eog = self.extract_features(self.EOGFeatureExtractor, mod2)
            utterance_emg = self.extract_features(self.EMGFeatureExtractor, mod3)
            utterance_mod2 = utterance_eog
            utterance_mod3 = utterance_emg

        elif self.config.dataset_name == 'HCI':
            utterance_ecg = self.extract_features(self.ECGFeatureExtractor, mod2)
            utterance_gsr = self.extract_features(self.GSRFeatureExtractor, mod3)
            utterance_mod2 = utterance_ecg
            utterance_mod3 = utterance_gsr

        elif self.config.dataset_name == 'MER':
            print_shape("Input mod2 (GSR)", mod2)
            print_shape("Input mod3 (PPG)", mod3)
            utterance_gsr = self.extract_features(self.GSRFeatureExtractor, mod2)
            utterance_ppg = self.extract_features(self.PPGFeatureExtractor, mod3)
            utterance_mod2 = utterance_gsr
            utterance_mod3 = utterance_ppg

        else:
            raise ValueError('Invalid dataset name')

        print_shape("utterance_mod2", utterance_mod2)
        print_shape("utterance_mod3", utterance_mod3)

        # Shared-private encoders
        self.shared_private(utterance_eeg, utterance_mod2, utterance_mod3)
        print_shape("After shared_private - utt_shared_eeg", self.utt_shared_eeg)
        print_shape("After shared_private - utt_shared_eog", self.utt_shared_eog)
        print_shape("After shared_private - utt_shared_emg", self.utt_shared_emg)

        # 根据数据集选择 reshape 策略
        if self.config.dataset_name == 'MER':
            # MER: (batch, 64) -> 需要变成4D用于FFT和卷积
            # 先尝试 reshape 为 (batch, 16, 2, 2)，然后插值上采样
            target_size = (32, 64)

            def mer_reshape_and_interpolate(tensor, name):
                print_shape(f"{name} before reshape", tensor)
                # (batch, 64) -> (batch, 16, 2, 2)
                t = tensor.view(batch_size, 16, 2, 2)
                print_shape(f"{name} after view(16,2,2)", t)
                # 插值到 (batch, 16, 32, 64)
                t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
                print_shape(f"{name} after interpolate", t)
                return t

            self.utt_shared_eeg = mer_reshape_and_interpolate(self.utt_shared_eeg, "utt_shared_eeg")
            self.utt_shared_eog = mer_reshape_and_interpolate(self.utt_shared_eog, "utt_shared_eog")
            self.utt_shared_emg = mer_reshape_and_interpolate(self.utt_shared_emg, "utt_shared_emg")
            self.utt_private_eeg = mer_reshape_and_interpolate(self.utt_private_eeg, "utt_private_eeg")
            self.utt_private_eog = mer_reshape_and_interpolate(self.utt_private_eog, "utt_private_eog")
            self.utt_private_emg = mer_reshape_and_interpolate(self.utt_private_emg, "utt_private_emg")
            self.utt_eeg_orig = mer_reshape_and_interpolate(self.utt_eeg_orig, "utt_eeg_orig")
            self.utt_eog_orig = mer_reshape_and_interpolate(self.utt_eog_orig, "utt_eog_orig")
            self.utt_emg_orig = mer_reshape_and_interpolate(self.utt_emg_orig, "utt_emg_orig")

        else:  # DEAP, HCI
            self.utt_shared_eeg = self.utt_shared_eeg.view(batch_size, 16, 32, 64)
            self.utt_shared_eog = self.utt_shared_eog.view(batch_size, 16, 32, 64)
            self.utt_shared_emg = self.utt_shared_emg.view(batch_size, 16, 32, 64)
            self.utt_private_eeg = self.utt_private_eeg.view(batch_size, 16, 32, 64)
            self.utt_private_eog = self.utt_private_eog.view(batch_size, 16, 32, 64)
            self.utt_private_emg = self.utt_private_emg.view(batch_size, 16, 32, 64)
            self.utt_eeg_orig = self.utt_eeg_orig.view(batch_size, 16, 32, 64)
            self.utt_eog_orig = self.utt_eog_orig.view(batch_size, 16, 32, 64)
            self.utt_emg_orig = self.utt_emg_orig.view(batch_size, 16, 32, 64)

        # FFT processing
        print_shape("utt_shared_eeg before FFT", self.utt_shared_eeg) #(B, 16, 32, 64)
        self.utt_shared_eeg_fft = torch.fft.fft(self.utt_shared_eeg.contiguous(), dim=-1)
        self.utt_shared_eeg_fft = torch.abs(self.utt_shared_eeg_fft)
        print_shape("utt_shared_eeg_fft", self.utt_shared_eeg_fft) #(B, 16, 32, 64)

        self.utt_shared_eog_fft = torch.fft.fft(self.utt_shared_eog, dim=-1)
        self.utt_shared_eog_fft = torch.abs(self.utt_shared_eog_fft)

        self.utt_shared_emg_fft = torch.fft.fft(self.utt_shared_emg, dim=-1)
        self.utt_shared_emg_fft = torch.abs(self.utt_shared_emg_fft)

        # Conv processing
        print_shape("utt_shared_eeg_fft before eeg_conv", self.utt_shared_eeg_fft) #(B, 16, 32, 64)
        self.utt_shared_eeg_conv = self.eeg_conv(self.utt_shared_eeg_fft)
        print_shape("utt_shared_eeg_conv after eeg_conv", self.utt_shared_eeg_conv) #(B, 16, 1, 33)
        self.utt_shared_eeg_conv = self.bn1(self.utt_shared_eeg_conv)
        eeg_conv_flat = self.utt_shared_eeg_conv.view(batch_size, -1)
        print_shape("eeg_conv_flat", eeg_conv_flat) #(B, 528)

        # 如果是MER且第一次运行，重新创建leaner层
        if self.config.dataset_name == 'MER' and not self._leaner_initialized:
            eeg_conv_size = eeg_conv_flat.size(1)
            print(f"\n[INIT] Creating leaner with input_size={eeg_conv_size}")
            self.leaner = nn.Linear(eeg_conv_size, 64).to(eeg_conv_flat.device)
            self._leaner_initialized = True

        self.utt_shared_eeg_conv = eeg_conv_flat
        print_shape("utt_shared_eeg_conv flattened", self.utt_shared_eeg_conv)

        self.utt_shared_eog_conv = self.e_com_g_conv(self.utt_shared_eog_fft)
        self.utt_shared_eog_conv = self.bn1(self.utt_shared_eog_conv)
        eog_conv_flat = self.utt_shared_eog_conv.view(batch_size, -1)
        print_shape("eog_conv_flat", eog_conv_flat)

        # 同样处理leaner_1
        if self.config.dataset_name == 'MER' and not hasattr(self, '_leaner_1_initialized'):
            eog_conv_size = eog_conv_flat.size(1)
            print(f"\n[INIT] Creating leaner_1 with input_size={eog_conv_size}")
            self.leaner_1 = nn.Linear(eog_conv_size, 64).to(eog_conv_flat.device)
            self._leaner_1_initialized = True

        self.utt_shared_eog_conv = eog_conv_flat

        self.utt_shared_emg_conv = self.e_com_g_conv(self.utt_shared_emg_fft)
        self.utt_shared_emg_conv = self.bn1(self.utt_shared_emg_conv)
        emg_conv_flat = self.utt_shared_emg_conv.view(batch_size, -1)
        print_shape("emg_conv_flat", emg_conv_flat)

        # 同样处理leaner_2
        if self.config.dataset_name == 'MER' and not hasattr(self, '_leaner_2_initialized'):
            emg_conv_size = emg_conv_flat.size(1)
            print(f"\n[INIT] Creating leaner_2 with input_size={emg_conv_size}")
            self.leaner_2 = nn.Linear(emg_conv_size, 64).to(emg_conv_flat.device)
            self._leaner_2_initialized = True

        self.utt_shared_emg_conv = emg_conv_flat

        # leaner processing
        print_shape("utt_shared_eeg_conv before leaner", self.utt_shared_eeg_conv) #(B, 528)
        self.utt_shared_eeg_1 = self.leaner(self.utt_shared_eeg_conv)
        print_shape("utt_shared_eeg_1 after leaner", self.utt_shared_eeg_1) #(B, 64)

        print_shape("utt_shared_eog_conv before leaner_1", self.utt_shared_eog_conv)
        self.utt_shared_eog_1 = self.leaner_1(self.utt_shared_eog_conv)
        print_shape("utt_shared_eog_1 after leaner_1", self.utt_shared_eog_1)

        self.utt_shared_emg_1 = self.leaner_2(self.utt_shared_emg_conv)

        # private branch processing
        self.utt_private_eeg = self.eeg_conv(self.utt_private_eeg)
        self.utt_private_eeg = self.bn1(self.utt_private_eeg)
        self.utt_private_eeg = self.utt_private_eeg.view(batch_size, -1)
        print_shape("utt_private_eeg after conv", self.utt_private_eeg)

        self.utt_private_eog = self.e_com_g_conv(self.utt_private_eog)
        self.utt_private_eog = self.bn1(self.utt_private_eog)
        self.utt_private_eog = self.utt_private_eog.view(batch_size, -1)

        self.utt_private_emg = self.e_com_g_conv(self.utt_private_emg)
        self.utt_private_emg = self.bn1(self.utt_private_emg)
        self.utt_private_emg = self.utt_private_emg.view(batch_size, -1)

        self.utt_private_eeg_1 = self.layer_norm(self.leaner(self.utt_private_eeg))
        self.utt_private_eog_1 = self.layer_norm(self.leaner_1(self.utt_private_eog))
        self.utt_private_emg_1 = self.layer_norm(self.leaner_2(self.utt_private_emg))

        # orig branch
        self.utt_eeg_orig = self.eeg_conv(self.utt_eeg_orig)
        self.utt_eeg_orig = self.utt_eeg_orig.view(batch_size, -1)
        self.utt_eog_orig = self.e_com_g_conv(self.utt_eog_orig)
        self.utt_eog_orig = self.utt_eog_orig.view(batch_size, -1)
        self.utt_emg_orig = self.e_com_g_conv(self.utt_emg_orig)
        self.utt_emg_orig = self.utt_emg_orig.view(batch_size, -1)

        self.utt_eeg_orig_1 = self.leaner(self.utt_eeg_orig)
        self.utt_eog_orig_1 = self.leaner_1(self.utt_eog_orig)
        self.utt_emg_orig_1 = self.leaner_2(self.utt_emg_orig)

        # GD-Unit
        self.prepare_for_HOMO_GD(self.utt_shared_eeg_1, self.utt_shared_eog_1, self.utt_shared_emg_1)
        self.prepare_for_HEME_GD(self.utt_private_eeg_1, self.utt_private_eog_1, self.utt_private_emg_1)

        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_eeg_1, self.utt_private_eog_1, self.utt_private_emg_1,
                         self.utt_shared_eeg_1, self.utt_shared_eog_1, self.utt_shared_emg_1), dim=0)
        print_shape("Stacked for transformer", h) #(6, B, 64)
        h = self.transformer_encoder(h)
        print_shape("After transformer", h) #(6, B, 64)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        print_shape("After concat", h) #(B, 384)
        h = self.utt_feature_norm(h)
        self.utt_subject = h

        self.utt_shared_subject = self.subject_shared(h) #(B, 384)

        h = h.reshape(batch_size, 1, -1)
        self.utt_private_subject = torch.ones_like(h) #(B, 1, 384)

        for i in range(batch_size):
            subject_id = int(subject_labels[i].item())
            for k, group in enumerate(groups, start=0):
                if subject_id in group:
                    self.utt_private_subject[i] = self.subject_private[k - 1](h[i])
                    break

        self.utt_private_subject = self.utt_private_subject.reshape(batch_size, -1)
        self.utt_private_subject = self.layer_norm_1(self.utt_private_subject) #(B, 384)

        h = h.reshape(self.utt_shared_subject.shape[0], self.utt_shared_subject.shape[1])

        o = self.fusion(self.utt_shared_subject + h + self.utt_private_subject)
        print_shape("Final output", o)

        return o

    def shared_private(self, utterance_eeg, utterance_eog, utterance_emg):
        print_shape("shared_private input utterance_eeg", utterance_eeg) #(B, 16)

        # Projecting to same space
        self.utt_eeg_orig = utterance_eeg = self.project_eeg(utterance_eeg)
        print_shape("After project_eeg", utterance_eeg) #(B, 64)

        self.utt_eog_orig = utterance_eog = self.project_eog(utterance_eog)
        self.utt_emg_orig = utterance_emg = self.project_emg(utterance_emg)

        # Private-shared components
        self.utt_private_eeg = self.private_eeg(utterance_eeg) #(B, 64)
        self.utt_private_eog = self.private_eog(utterance_eog)
        self.utt_private_emg = self.private_emg(utterance_emg)

        self.utt_shared_eeg = self.shared(utterance_eeg) #(B, 64)
        self.utt_shared_eog = self.shared(utterance_eog)
        self.utt_shared_emg = self.shared(utterance_emg)

    def prepare_for_HOMO_GD(self, utt_shared_eeg, utt_shared_eog, utt_shared_emg):
        self.repr_eeg_low = self.proj1_eeg_low(utt_shared_eeg)
        hs_proj_eeg_low = self.proj2_eeg_low(
            F.dropout(F.relu(self.repr_eeg_low, inplace=True), p=0.5, training=True))
        hs_proj_eeg_low += utt_shared_eeg
        self.logits_eeg_low = self.out_layer_eeg_low(hs_proj_eeg_low)

        self.repr_mod2_low = self.proj1_mod2_low(utt_shared_eog)
        hs_proj_mod2_low = self.proj2_mod2_low(
            F.dropout(F.relu(self.repr_mod2_low, inplace=True), p=0.5, training=True))
        hs_proj_mod2_low += utt_shared_eog
        self.logits_mod2_low = self.out_layer_mod2_low(hs_proj_mod2_low)

        self.repr_mod3_low = self.proj1_mod3_low(utt_shared_emg)
        hs_proj_mod3_low = self.proj2_mod3_low(
            F.dropout(F.relu(self.repr_mod3_low, inplace=True), p=0.5, training=True))
        hs_proj_mod3_low += utt_shared_emg
        self.logits_mod3_low = self.out_layer_mod3_low(hs_proj_mod3_low)

    def prepare_for_HEME_GD(self, utt_shared_eeg, utt_shared_eog, utt_shared_emg):
        self.repr_eeg_high = self.proj1_eeg_high(utt_shared_eeg)
        hs_proj_eeg_high = self.proj2_eeg_high(
            F.dropout(F.relu(self.repr_eeg_high, inplace=True), p=0.5, training=True))
        hs_proj_eeg_high += utt_shared_eeg
        self.logits_eeg_high = self.out_layer_eeg_high(hs_proj_eeg_high)

        self.repr_mod2_high = self.proj1_mod2_high(utt_shared_eog)
        hs_proj_mod2_high = self.proj2_mod2_high(
            F.dropout(F.relu(self.repr_mod2_high, inplace=True), p=0.5, training=True))
        hs_proj_mod2_high += utt_shared_eog
        self.logits_mod2_high = self.out_layer_mod2_high(hs_proj_mod2_high)

        self.repr_mod3_high = self.proj1_mod3_high(utt_shared_emg)
        hs_proj_mod3_high = self.proj2_mod3_high(
            F.dropout(F.relu(self.repr_mod3_high, inplace=True), p=0.5, training=True))
        hs_proj_mod3_high += utt_shared_emg
        self.logits_mod3_high = self.out_layer_mod3_high(hs_proj_mod3_high)

    def forward(self, eeg, eog, emg, lengths, subject_labels, groups):
        o = self.alignment(eeg, eog, emg, lengths, subject_labels, groups)
        return o