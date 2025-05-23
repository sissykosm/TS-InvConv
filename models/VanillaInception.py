import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.InvConvolution import InvConv1d, InceptionInvConv, MultiScaleInvConv, SparseMultiScaleConv
from typing import cast, Union, List

### Vanilla Inception ###
########################################################################################
#SOURCE: https://github.com/TheMrGhostman/InceptionTime-Pytorch/blob/master/inception.py
########################################################################################

class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)

class InceptionModel(nn.Module):

    def __init__(self, num_blocks, in_channels, out_channels,
                 bottleneck_channels, kernel_sizes,
                 use_residuals='default',
                 num_pred_classes=1
                 ):
        super().__init__()

        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes
        }

        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels, num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(cast(Union[bool, List[bool]], use_residuals), num_blocks))

        self.layer1 = nn.Sequential(*[
            InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                           residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                           kernel_size=kernel_sizes[i]) for i in range(num_blocks)
        ])

    @staticmethod
    def _expand_to_blocks(value, num_blocks):
        if isinstance(value, list):
            assert len(value) == num_blocks
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x):
        x = self.layer1(x)
        return x


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 residual, stride=1, bottleneck_channels=32,
                 kernel_size=41):
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size=1, bias=False)

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=stride, bias=False)
            for i in range(len(kernel_size_s))
        ])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    def forward(self, x):
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in

        self.model = InceptionModel(num_blocks=3, in_channels=self.in_channels, out_channels=32, bottleneck_channels=32, kernel_sizes=[10, 20, 40], use_residuals=True)
        
        if self.task_name == 'anomaly_detection':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(32, configs.c_out, bias=True)
        if self.task_name == 'classification' or self.task_name == 'classification_pt':
            self.GAP = nn.AvgPool1d(self.seq_len)
            self.projection = nn.Sequential(nn.Linear(32, configs.num_class))
    
    def anomaly_detection(self, x_enc):
        enc_out = self.model(x_enc.permute(0, 2, 1))
        dec_out = self.predict_linear(enc_out)
        dec_out = self.projection(dec_out.permute(0, 2, 1))
        return dec_out
    
    def classification(self, x_enc):
        enc_out = self.model(x_enc.permute(0, 2, 1)).mean(dim=-1)
        dec_out = self.projection(enc_out)
        return dec_out
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'classification' or 'classification_pt':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
