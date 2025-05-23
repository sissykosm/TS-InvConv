import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.InvConvolution import InvConv1d, InceptionInvConv, MultiScaleInvConv, SparseMultiScaleConv

### Vanilla cnn ###
class ConvNet(nn.Module):
    def __init__(self,original_length,original_dim):
        super(ConvNet, self).__init__()
    
        self.kernel_size = 3
        self.padding = 1

        self.layer1 = nn.Sequential(
                nn.Conv1d(original_dim, 64, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                )
                
        self.layer2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                )
        
        self.layer21 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                )

        self.layer22 = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                )

        self.layer23 = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                )

        self.layer3 = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=self.kernel_size, padding=self.padding),
                nn.ReLU(),
                )
                          
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer21(out)
        out = self.layer22(out)
        out = self.layer3(out)

        return out

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in

        self.model = ConvNet(self.seq_len, self.in_channels)
        
        if self.task_name == 'anomaly_detection':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len+self.seq_len)
            self.projection = nn.Linear(
                256, configs.c_out, bias=True)
        if self.task_name == 'classification' or self.task_name == 'classification_pt':
            self.GAP = nn.AvgPool1d(self.seq_len)
            self.projection = nn.Sequential(nn.Linear(256, configs.num_class)) 

    def anomaly_detection(self, x_enc):
        enc_out = self.model(x_enc.permute(0,2,1))
        dec_out = self.predict_linear(enc_out)
        dec_out = self.projection(dec_out.permute(0, 2, 1))
        return dec_out

    def classification(self, x_enc):
        enc_out = self.model(x_enc.permute(0,2,1))

        enc_out = self.GAP(enc_out)

        enc_out = enc_out.reshape(enc_out.size(0), -1)
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
