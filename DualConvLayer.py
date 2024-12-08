'''
This script is used to achieve the local feature extraction within the modality. 

Three convolutional blocks are utilized respectively in the temporal and spatial dimensions of EEG and fNIRS.

'''

import numpy as np
import torch
from torch import nn as nn

channel_seq = (np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 13, 17, 18, 21, 23, 24, 27, 29, 30, 33, 35, 36, 39, 16, 20, 19, 22,
     26, 25, 28, 32, 31, 34, 38, 37, 40]) - 1).tolist()

class SWConv2d(nn.Module):
    '''
    Inspired by EEGNet.

    SW_conv = Depthwise_conv + Pointwise_conv

    '''
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), stride=(1, 1), bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                    groups=in_channels, stride=stride, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def channel_shuffle(self, x):
        groups = self.in_channels
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # grouping, 通道分组
        # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # channel shuffle, 通道洗牌
        x = torch.transpose(x, 1, 2).contiguous()
        # x.shape=(batch_size, channels_per_group, groups, height, width)
        # flatten
        x = x.view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        # x = self.channel_shuffle(x)

        return x


class EEGSpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, bias=False):
        self.dropout = dropout
        super().__init__()

        # emb_size= = 64
        # outputs_size = (60, 1000 / 4)
        pooling_kernel = [4, 2, 5]
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (60, 250 / 2)
        self.eeg_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (60, 125 / 5)
        self.eeg_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

        # outputs_size = (60, 1)
        self.temporal_pooling = nn.AdaptiveAvgPool2d((60, 1))

    def forward(self, eeg):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)

        eeg = self.eeg_block1(eeg)
        eeg = self.eeg_block2(eeg)
        eeg = self.eeg_block3(eeg)
        outputs = self.temporal_pooling(eeg)
        return outputs


class NIRSSpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, bias=False):
        super().__init__()

        # emb_size = 64
        # outputs_size = (40, 40 / 2)
        pooling_kernel = [2, 1, 2]
        self.dropout = dropout
        self.nirs_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (40, 20 / 1)
        self.nirs_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (40, 20 / 2)
        self.nirs_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

        # outputs_size = (40, 1)
        self.temporal_pooling = nn.AdaptiveAvgPool2d((40, 1))

    def forward(self, nirs):
        nirs = nirs[:, channel_seq, :]
        if nirs.ndim == 3:
            nirs = nirs.unsqueeze(1)
        nirs = self.nirs_block1(nirs)
        nirs = self.nirs_block2(nirs)
        nirs = self.nirs_block3(nirs)
        outputs = self.temporal_pooling(nirs)

        return outputs


class EEGTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, bias=False):
        self.dropout = dropout
        super().__init__()


        # emb_size = 64
        # outputs_size = (60, 1000 / 4)
        pooling_kernel = [4, 1, 5]
        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (1, 250 / 1)
        self.eeg_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(60, 1), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (1, 250 / 5)
        self.eeg_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 15), padding=(0, 15 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, eeg):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)

        eeg = self.eeg_block1(eeg)
        eeg = self.eeg_block2(eeg)
        eeg = self.eeg_block3(eeg)
        return eeg


class NIRSTemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout, bias=False):
        super().__init__()
        
        # emb_size = 64
        # outputs_size = (40, 40 / 2)
        pooling_kernel = [2, 1, 2]
        self.dropout = dropout

        # outputs_size = (40, 40 / 2)
        self.nirs_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=emb_size // 1, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[0])),
        )

        # outputs_size = (1, 20 / 1)
        self.nirs_block2 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size // 1, kernel_size=(40, 1), padding=(0, 0), bias=bias),
            nn.BatchNorm2d(emb_size // 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[1])),
        )

        # outputs_size = (1, 20 / 2)
        self.nirs_block3 = nn.Sequential(
            SWConv2d(in_channels=emb_size // 1, out_channels=emb_size, kernel_size=(1, 3), padding=(0, 3 // 2), bias=bias),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.AvgPool2d((1, pooling_kernel[2])),
        )

    def forward(self, nirs):
        nirs = nirs[:, channel_seq, :]
        if nirs.ndim == 3:
            nirs = nirs.unsqueeze(1)
        nirs = self.nirs_block1(nirs)
        nirs = self.nirs_block2(nirs)
        nirs = self.nirs_block3(nirs)

        return nirs


class SpatialConvLayer(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.eeg_spatial_projection = EEGSpatialConvLayer(emb_size, dropout, bias=True)
        self.nirs_spatial_projection = NIRSSpatialConvLayer(emb_size, dropout, bias=True)

    def forward(self, eeg, nirs):
        spatial_eeg_features = self.eeg_spatial_projection(eeg)
        spatial_nirs_features = self.nirs_spatial_projection(nirs)

        return spatial_eeg_features, spatial_nirs_features



class TemporalConvLayer(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.eeg_temporal_projection = EEGTemporalConvLayer(emb_size, dropout)
        self.nirs_temporal_projection = NIRSTemporalConvLayer(emb_size, dropout)

    def forward(self, eeg, nirs):
        temporal_eeg_features = self.eeg_temporal_projection(eeg)
        temporal_nirs_features = self.nirs_temporal_projection(nirs)

        return temporal_eeg_features, temporal_nirs_features