import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.InvConvolution import InvConv1d, InceptionInvConv, MultiScaleInvConv, SparseMultiScaleConv
import numpy as np

def split_power_of_2(value):
    assert value % 2 == 0, "Value must be divisible by 2"
    
    # Step 1: Split value into two equal parts (slightly adjusted for rounding to the same decade)
    equal_part = (value // 3) // 2 * 2  # Keep equal parts divisible by 2
    
    # Step 2: Calculate the remainder (which will be larger than the equal parts)
    larger_part = value - 2 * equal_part
    
    # Step 3: Ensure the larger part is still divisible by 2 and aligns with the same decade
    return larger_part, equal_part, equal_part

def get_kernel_size(length):
    powers = np.floor(np.log2(length//2))
    if powers>=4:
        return 2**np.arange(4,min(powers+1,8))[::-1]
    else:
        return [np.floor(length//2)]

### Invariant cnns ###
class InceptionInvConvNet(nn.Module):
    def __init__(self,original_length,original_dim,hidden_dims,init_type='gaussian'):
        super(InceptionInvConvNet, self).__init__()
        
        self.kernel_sizes = [51,75,101,125]
        self.hidden_dims = hidden_dims

        self.layer1 = InceptionInvConv(in_channels=original_dim, out_channels=self.hidden_dims, kernel_sizes=self.kernel_sizes, init_type=init_type)
        self.bn = nn.BatchNorm1d((self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2])*len(self.kernel_sizes))
        self.relu = nn.ReLU()
            
    def forward(self, x):
        out, coef_matrix = self.layer1(x)  
        out = self.bn(out)
        out = self.relu(out)         
        return out, coef_matrix

class InvConvNet_4(nn.Module):
    def __init__(self,original_length,original_dim,hidden_dims,init_type='gaussian'):
        super(InvConvNet_4, self).__init__()
        self.hidden_dims = hidden_dims
        self.kernel_size = min(50,original_length//2)

        if min(50,original_length//2)%2==0:
            self.kernel_size = self.kernel_size + 1
        self.padding = self.kernel_size//2

        self.layer1 = InvConv1d(original_dim,self.hidden_dims, kernel_size=self.kernel_size, padding=self.padding, init_type=init_type) #(20,20,24)
        self.bn = nn.BatchNorm1d((self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2]))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, coef_matrix = self.layer1(x)
        
        out = self.bn(out)
        out = self.relu(out)
        return out, coef_matrix

class MultiscaleInvConvNet(nn.Module):
    def __init__(self,original_length,original_dim,hidden_dims,init_type='gaussian'):
        super(MultiscaleInvConvNet,self).__init__()
        self.hidden_dims = hidden_dims
        self.kernel_sizes = get_kernel_size(original_length)
        print('Kernel sizes', self.kernel_sizes)
        self.layer1 = MultiScaleInvConv(in_channels=original_dim, out_channels=self.hidden_dims, kernel_sizes=self.kernel_sizes,init_type="cosine")
        self.bn1 = nn.BatchNorm1d((self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2])*len(self.kernel_sizes))
        self.relu1 = nn.ReLU()

        self.layer2 = SparseMultiScaleConv([self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2]]*len(self.kernel_sizes),self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2],kernel_sizes=self.kernel_sizes,bias=True,init_type="gaussian")
        self.bn2 = nn.BatchNorm1d(self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2])
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out, coef_matrix = self.layer1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out, coef_matrix

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in

        self.model_type = configs.invconv_type #invconv, invinception, multiscaleinvconv
        print('Model', self.model_type)

        larger_part, part1, part2 = split_power_of_2(configs.d_model)
        self.hidden_dims = (larger_part, part1, part2)
        
        if configs.inv_ablation==1:
            self.hidden_dims = (configs.d_model, 0, 0)
        elif configs.inv_ablation==2:
            self.hidden_dims = (0, configs.d_model, 0)
        elif configs.inv_ablation==3:
            self.hidden_dims = (0, 0, configs.d_model)

        print('Hidden', self.hidden_dims)

        if self.model_type == "invconv":
            self.model = InvConvNet_4(self.seq_len, self.in_channels, self.hidden_dims) 
        elif self.model_type == "invinception":
            self.model = InceptionInvConvNet(self.seq_len, self.in_channels, self.hidden_dims) 
        else:
            self.model = MultiscaleInvConvNet(self.seq_len, self.in_channels, self.hidden_dims) 
        
        if self.task_name == 'anomaly_detection':
            self.kernel_sizes = get_kernel_size(self.seq_len)
            self.num_kernels = len(self.kernel_sizes)
            
            self.num_channels = 5  

            self.predict_linear = nn.ModuleList([nn.Linear(int(self.seq_len-max(self.kernel_sizes)+2),self.pred_len + self.seq_len)]+[nn.Linear(self.seq_len+1, self.pred_len + self.seq_len) for _ in range(self.num_kernels * self.num_channels)])
            self.pr_n = nn.ModuleList([nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1) for _ in range(self.num_kernels*2)]) #self.num_kernels
            
            self.project = nn.ModuleList([
                nn.Conv1d(self.hidden_dims[0], self.in_channels, kernel_size=1),
                nn.Conv1d(self.hidden_dims[1], self.in_channels, kernel_size=1),
                nn.Conv1d(self.hidden_dims[2], self.in_channels, kernel_size=1),
            ] * self.num_kernels)
            self.bn = nn.ModuleList([nn.LayerNorm(self.in_channels) for _ in range(self.num_kernels*3)])
            self.relu = nn.ModuleList([nn.ReLU() for _ in range(self.num_kernels*3)])
            self.dropout = nn.ModuleList([nn.Dropout(p=0.5) for _ in range(self.num_kernels*3)])  # Dropout layer added

            self.projection = nn.ModuleList([nn.Conv1d(self.in_channels, configs.c_out, kernel_size=1) for _ in range(self.num_kernels*3)])
        if self.task_name == 'classification' or self.task_name == 'classification_pt':
            self.dropout = nn.Dropout(configs.dropout)
            if self.model_type == "invconv" or self.model_type == "invinception":
                self.GAP = nn.AvgPool1d(self.seq_len)
            else:
                self.GAP = nn.AvgPool1d(int(self.seq_len-max(self.kernel_sizes)+2))
            
            if self.model_type == "invinception":
                self.projection = nn.Sequential(nn.Linear((self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2])*4, configs.num_class))
            else:
                self.projection = nn.Sequential(nn.Linear((self.hidden_dims[0]+self.hidden_dims[1]+self.hidden_dims[2]), configs.num_class)) #*4 for inceptioninv

    def anomaly_detection(self, x_enc):
        enc_out, coef_matrix = self.model(x_enc.permute(0, 2, 1))
        #dec_out = self.projection(enc_out.permute(0,2,1))
        
        z_predict = self.predict_linear[0](enc_out)

        hs = self.hidden_dims[0] + self.hidden_dims[1] + self.hidden_dims[2]
        dec_out = 0
        for i in range(self.num_kernels):
            dec_out_part = 0
            n1 = self.predict_linear[1 + i * self.num_channels](self.pr_n[2*i](coef_matrix[:, i, 0, :, :])) #*2 + 1
            c1 = self.predict_linear[2 + i * self.num_channels](coef_matrix[:, i, 1, :, :])
            n2 = self.predict_linear[3 + i * self.num_channels](self.pr_n[2*i+1](coef_matrix[:, i, 2, :, :]))
            c2 = self.predict_linear[4 + i * self.num_channels](coef_matrix[:, i, 3, :, :])
            c3 = self.predict_linear[5 + i * self.num_channels](coef_matrix[:, i, 4, :, :])

            if self.hidden_dims[0]!=0:
                zA = self.project[i * 3](z_predict[:, 0 * hs:0 * hs + self.hidden_dims[0], :])
                zA = self.dropout[3*i](self.relu[3*i](self.bn[3*i](zA.permute(0,2,1)).permute(0,2,1)))
                dec_out_part += self.projection[i * 3](zA).permute(0, 2, 1)

            if self.hidden_dims[1]!=0:
                zB = self.project[i * 3 + 1](z_predict[:, 0 * hs + self.hidden_dims[0]:0 * hs + self.hidden_dims[0] + self.hidden_dims[1], :]) * n1
                zB = self.dropout[3 * i + 1](self.relu[3*i+1](self.bn[3*i+1](zB.permute(0,2,1)).permute(0,2,1)))
                dec_out_part += self.projection[i * 3 + 1](zB).permute(0, 2, 1)
            
            if self.hidden_dims[2]!=0:
                zC = self.project[i * 3 + 2](z_predict[:, 0 * hs + self.hidden_dims[0] + self.hidden_dims[1]: (0 + 1) * hs, :]) * n2
                zC = self.dropout[3 * i + 2](self.relu[3*i+2](self.bn[3*i+2](zC.permute(0,2,1)).permute(0,2,1)))
                dec_out_part += self.projection[i * 3 + 2](zC).permute(0, 2, 1)

            dec_out_part += (c1 + c2 + c3).permute(0, 2, 1) 
            dec_out += dec_out_part
        
        return dec_out

    def classification(self, x_enc):
        enc_out = self.model(x_enc.permute(0,2,1))[0] 

        enc_out = self.dropout(enc_out)
        feature_maps = enc_out
        enc_out = self.GAP(enc_out)
        
        enc_out = enc_out.reshape(enc_out.size(0), -1)
        dec_out = self.projection(enc_out)
        return dec_out, feature_maps

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'classification' or self.task_name == 'classification_pt':
            dec_out, feature_maps = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
