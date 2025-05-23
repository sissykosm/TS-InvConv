import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.InvConvolution import InvConv1d, InceptionInvConv, MultiScaleInvConv, SparseMultiScaleConv

### Vanilla Resnet ###
########################################################################################
#SOURCE: https://github.com/okrasolar/pytorch-timeseries/blob/master/
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
        
    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride, padding=padding // 2, dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),)

    def forward(self, x):
        return self.layers(x)


class ResNetBaseline(nn.Module):
        
    def __init__(self, in_channels, mid_channels = 64, task='anomaly_detection'):
        super().__init__()
        self.task = task
        self.input_args = {
            'in_channels': in_channels}

        self.layer1 = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),])

    def forward(self, x):
        x = self.layer1(x)
        if self.task!='anomaly_detection':
            return x.mean(dim=-1)
        else:
            return x


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)])

    def forward(self, x):
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in

        self.model = ResNetBaseline(self.in_channels, mid_channels=64, task=self.task_name)

        if self.task_name == 'anomaly_detection':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                64*2, configs.c_out, bias=True)
        if self.task_name == 'classification' or self.task_name == 'classification_pt':
            self.GAP = nn.AvgPool1d(self.seq_len)
            self.projection = nn.Sequential(nn.Linear(64*2, configs.num_class))  

    def anomaly_detection(self, x_enc):
        enc_out = self.model(x_enc.permute(0, 2, 1))
        dec_out = self.predict_linear(enc_out)
        dec_out = self.projection(dec_out.permute(0, 2, 1))
        return dec_out

    def classification(self, x_enc):
        enc_out = self.model(x_enc.permute(0, 2, 1))
        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'classification' or self.task_name == 'classification_pt':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None

