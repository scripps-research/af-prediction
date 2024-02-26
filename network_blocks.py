import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Module):

    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 **kwargs,
                 ):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        pad = self.calc_same_pad(i=x.size()[-1], k=self.kernel_size, s=self.stride, d=self.dilation)
        x = F.pad(x, [pad // 2, pad - pad // 2])
        return self.conv(x)


class ConvLayer(nn.Module):

    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size,
                #  stride=1,
                 batch_norm=True,
                 **kwargs,
                 ):
        super().__init__()

        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs, bias=False if batch_norm else True))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        # if stride > 1:
        #     layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x


class ResLayer(nn.Module):

        
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size,
                 batch_norm=True,
                 stride=1,
                 **kwargs,
                 ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        layers = []

        # Main branch
        layers.append(Conv1dSamePadding(in_channels, out_channels, kernel_size, stride=stride, **kwargs, bias=False if batch_norm else True))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        layers.append(Conv1dSamePadding(out_channels, out_channels, kernel_size, **kwargs, bias=False if batch_norm else True))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        # Side branch
        side_layers = []
        if stride != 1 or in_channels != out_channels:
            side_layers.append(Conv1dSamePadding(in_channels, out_channels, kernel_size=1, stride=stride, bias=False if batch_norm else True))
            if batch_norm:
                side_layers.append(nn.BatchNorm1d(out_channels))
        self.side_layers = nn.Sequential(*side_layers)
        
    def forward(self, x):
        if len(self.side_layers) == 0:
            return self.layers(x) + x
        return self.layers(x) + self.side_layers(x)


class ECGSplitProcessing(nn.Module):

    def __init__(self,
                 raw_ecg_stem_out_channels,
                 raw_ecg_stem_kernel_size,
                 raw_ecg_stem_stride,
                 raw_ecg_stem_en_max_pooling,
                 raw_ecg_stem_max_pool_kernel_size,
                 raw_ecg_stem_max_pool_stride,
                 raw_ecg_layer,
                 raw_ecg_batch_norm,
                 raw_ecg_conv_kernel_size,
                 raw_ecg_conv_channels,
                 raw_ecg_conv_strides,
                 raw_ecg_conv_pooling_kernels,
                 raw_ecg_conv_pooling_strides,
                 ):
        super().__init__()
        
        if raw_ecg_layer == 'ConvLayer':
            Layer = ConvLayer
        elif raw_ecg_layer == 'ResLayer':
            Layer = ResLayer
        else:
            raise Exception(f'Raw ECG layer {raw_ecg_layer} not supported')
        layers = []
        # Stem
        in_channels = 1
        layers.append(Conv1dSamePadding(in_channels, raw_ecg_stem_out_channels, kernel_size=raw_ecg_stem_kernel_size, stride=raw_ecg_stem_stride, bias=False if raw_ecg_batch_norm else True))
        if raw_ecg_batch_norm:
            layers.append(nn.BatchNorm1d(raw_ecg_stem_out_channels))
        layers.append(nn.ReLU())
        if raw_ecg_stem_en_max_pooling:
            layers.append(nn.MaxPool1d(kernel_size=raw_ecg_stem_max_pool_kernel_size, stride=raw_ecg_stem_max_pool_stride))
        # Layers
        in_channels = raw_ecg_stem_out_channels
        for out_channels, kernel_size, stride, max_pool_kernel, max_pool_stride in zip(raw_ecg_conv_channels, raw_ecg_conv_kernel_size, raw_ecg_conv_strides, raw_ecg_conv_pooling_kernels, raw_ecg_conv_pooling_strides):
            layers.append(Layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, batch_norm=raw_ecg_batch_norm))
            if max_pool_stride > 1 or max_pool_kernel > 1:
                layers.append(nn.MaxPool1d(max_pool_kernel, stride=max_pool_stride))
            in_channels = out_channels
        # Average Pooling layer
        layers.append(nn.AdaptiveAvgPool1d(1))
        # Create sequential layer
        self.raw_ecg_processing = nn.Sequential(*layers)   

    def forward(self, raw_ecg):

        out = []
        for x in torch.split(raw_ecg, 11989, dim=2):
            out.append(self.raw_ecg_processing(x))
        out = torch.concat(out, dim=2)

        return out
