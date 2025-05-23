from functools import partial
from typing import Iterable, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)
    
def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    padding: Union[int, Iterable[int]] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args: in
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.

    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_,device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel.to(signal.device), offset.to(signal.device))[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = rfftn(signal_[:,None,::], dim=tuple(range(3, signal.ndim+1)))
    kernel_fr = rfftn(padded_kernel[None,::], dim=tuple(range(3, signal.ndim+1)))
    kernel_fr.imag *= -1
    output = irfftn(signal_fr*kernel_fr, dim=tuple(range(3, signal.ndim+1)))

    # Remove extra padded values
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1)),slice(0, output.size(2))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    return output


class Conv1d(nn.Module): 

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[int, Iterable[int]] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        bias: bool = True,
        init_type: str = "gaussian"
    ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.with_bias = bias
        self.init_type = init_type
      
        self.weight = nn.Parameter(self.weight_initialisation())
        self.bias = nn.Parameter(torch.randn(1,self.out_channels,1)) if bias else None 

    def weight_initialisation(self):
        if self.init_type == "gaussian":
            weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size)
        elif self.init_type == "cosine":
            w_lst = []
            time = torch.linspace(0,torch.pi,self.kernel_size).reshape(1,1,-1)
            in_c_vec = torch.ones(self.in_channels).reshape(1,-1,1)
            # manage number of out channels
            c = min(self.out_channels,self.kernel_size//2)
            if c == self.out_channels:
                out_c_vector = torch.arange(1,c+1).reshape(-1,1,1)
            else:
                out_c_vector = torch.arange(1,c+1).repeat(self.out_channels//c+1)[:n_out].reshape(-1,1,1)
            weight = torch.cos(time*in_c_vec*out_c_vector)
            weight += 0.05*torch.randn(self.out_channels, self.in_channels, self.kernel_size)
        elif self.init_type == "randomwalk":
            weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size)
            weight = torch.cumsum(weight, dim=-1)
        elif "randomcosine":
            time = torch.linspace(0,torch.pi,self.kernel_size).reshape(1,1,-1)
            c_vec = torch.randint(0,self.kernel_size//2,(self.out_channels,self.in_channels,1))
            weight = torch.cos(time*c_vec)
            weight += 0.05*torch.randn(self.out_channels, self.in_channels, self.kernel_size)
        else:
            raise ValueError("Initialization type incorrect")
        return weight   

    def forward(self, signal):

        #preprocessiong 
        tnk,ndim,length = self.weight.shape

        if signal.shape[1] != ndim: 
            raise ValueError("Kernel dimension and signal dimension do not match")

        #compute all cross
        product = fft_conv(
            signal,
            self.weight,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation
            )
        
        product= torch.sum(product,dim=2) 
        if self.with_bias: 
            return product + self.bias
        else: 
            return product

class InvConv1d(nn.Module): 

    def __init__(
        self,
        in_channels: int,
        out_channels: Tuple, #A: nornal, B offset, C linear"
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        bias: bool = (True,True,True),
        init_type: str = "gaussian"
    ):
        "Pas de groupes toujours 1d A: nornal, B offset, C linear"
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.bias_mask = bias
        self.bias_sizes =tuple(x*y for x,y in zip(out_channels,bias))
        self.init_type = init_type

        if len(out_channels) !=3:
            raise ValueError(
                    "'out_channels' must be a tuple of length 3."
                )
        
        if len(bias) !=3:
            raise ValueError(
                    "'bias' must be a tuple of length 3."
                )

        #weight = torch.randn(sum(out_channels), in_channels, kernel_size)
        #self.weight = nn.Parameter(weight)
        self.weight = nn.Parameter(self.weight_initialisation())
        self.bias = nn.Parameter(torch.randn(sum(self.bias_sizes))) if any(bias) else None #offset only for type A : normal

    def weight_initialisation(self):
        if self.init_type == "gaussian":
            weight = torch.randn(sum(self.out_channels), self.in_channels, self.kernel_size)
        elif self.init_type == "cosine":
            w_lst = []
            time = torch.linspace(0,torch.pi,self.kernel_size).reshape(1,1,-1)
            in_c_vec = torch.ones(self.in_channels).reshape(1,-1,1)
            for n_out in self.out_channels:
                # manage number of out channels
                c = min(n_out,self.kernel_size//2)
                if c == n_out:
                        out_c_vector = torch.arange(1,c+1).reshape(-1,1,1)
                else:
                        out_c_vector = torch.arange(1,c+1).repeat(n_out//c+1)[:n_out].reshape(-1,1,1)
                w = torch.cos(time*in_c_vec*out_c_vector)
                w_lst.append(w)
            weight = torch.concat(w_lst)
            weight += 0.05*torch.randn(sum(self.out_channels), self.in_channels, self.kernel_size)
        elif self.init_type == "randomwalk":
            weight = torch.randn(sum(self.out_channels), self.in_channels, self.kernel_size)
            weight = torch.cumsum(weight, dim=-1)
        elif "randomcosine":
            w_lst = []
            time = torch.linspace(0,torch.pi,self.kernel_size).reshape(1,1,-1)
            for n_out in self.out_channels:
                c_vec = torch.randint(0,self.kernel_size//2,(n_out,self.in_channels,1))
                w = torch.cos(time*c_vec)
                w_lst.append(w)
            weight = torch.concat(w_lst)
            weight += 0.05*torch.randn(sum(self.out_channels), self.in_channels, self.kernel_size)
        else:
            raise ValueError("Initialization type incorrect")
        return weight   

    def forward(self, signal):

        #preprocessiong 
        tnk,ndim,length = self.weight.shape
        
        if signal.shape[1] != ndim: 
            raise ValueError(
                    "Kernel dimension and signal dimension do not match"
                )
        #epsilon = torch.tensor(1e-5).to(signal.device)
        mean_operator = torch.ones((1,1,length),device=signal.device)/length
        means = fft_conv(signal,mean_operator,padding=self.padding,padding_mode=self.padding_mode,stride=self.stride,dilation=self.dilation)
        stds = torch.sqrt(torch.clamp(fft_conv(signal**2,mean_operator,padding=self.padding,padding_mode=self.padding_mode,stride=self.stride,dilation=self.dilation) -means**2,0))
        norms = (stds*torch.sqrt(torch.tensor([length]).to(signal.device))) #+ epsilon
        keep_norms = norms
        norms = torch.where(norms!=0.0, norms, torch.tensor(float('inf'), device=signal.device))
        #norms = torch.where(norms !=0, norms, torch.inf)

        linear_operator = torch.arange(0,length,dtype=torch.float,device=signal.device).reshape((1,1,length))
        linear_std = ((length**2-1)/12)**0.5
        linear_mean = (length-1)/2
        linear_coef = (fft_conv(signal,linear_operator,padding=self.padding,padding_mode=self.padding_mode,stride=self.stride,dilation=self.dilation)/length - linear_mean*means)/linear_std**2
        linear_offset = means - linear_coef*linear_mean
        linear_norms = torch.sqrt(torch.clamp(length*(stds**2 - linear_coef**2*linear_std**2),0)) #+ epsilon
        keep_linear_norms = linear_norms
        linear_norms = torch.where(linear_norms!=0.0, linear_norms, torch.tensor(float('inf'), device=signal.device))
        #linear_norms = torch.where(linear_norms!=0,linear_norms,torch.inf)

        #compute all cross
        product = fft_conv(
            signal,
            self.weight,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation)
        
        Pa,Pb,Pc = torch.split(product,self.out_channels,dim=1)
        Ka,Kb,Kc = torch.split(self.weight,self.out_channels)
        if any(self.bias_mask):
            Ba,Bb,Bc = torch.split(self.bias,self.bias_sizes)

        if len(Ka)!=0:
            Nk,Nc,length = Ka.shape
            Pa = torch.sum(Pa,dim=2) #.view((Nk,Pa.shape[-1]))
            if self.bias_mask[0]: 
                Pa = Pa +  Ba.view(Nk,1)
        else: 
            Pa =  Pa.view(Pa.shape[0],0,Pa.shape[-1])

        if len(Kb) !=0: 
            Nk,Nc,length = Kb.shape
            Pb = Pb - torch.sum(Kb,dim = -1).view(1,Nk,Nc,1)*means
            Pb = Pb/norms
            Pb = torch.sum(Pb,dim=2)
            if self.bias_mask[1]: 
                Pb = Pb + Bb.view(1,Nk,1)
        else: 
            Pb = Pb.view(Pb.shape[0],0,Pb.shape[-1])

        if len(Kc)!=0: 
            Nk,Nc,length = Kc.shape
            Pc = Pc - (torch.sum(Kc * linear_operator, dim=-1).view(1,Nk,Nc,1)*linear_coef + torch.sum(Kc,dim = -1).view(1,Nk,Nc,1)*linear_offset)
            Pc = Pc/linear_norms
            Pc = torch.sum(Pc,dim=2)
            if self.bias_mask[2]: 
                Pc = Pc + Bc.view(Nk,1)
        else: 
            Pc = Pc.view(Pc.shape[0],0,Pc.shape[-1])
        
        output = torch.vstack((Pa.permute(1,0,2),Pb.permute(1,0,2),Pc.permute(1,0,2))).permute(1,0,2)

        means_stds_ltn_ltc_lto = torch.concat((keep_norms,means,keep_linear_norms,linear_coef,linear_offset), dim=1).unsqueeze(1)

        return output, means_stds_ltn_ltc_lto


def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes


class InceptionInvConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], init_type="gaussian"):
        super(InceptionInvConv, self).__init__()
        self.conv_from_bottleneck = nn.ModuleList()  
        for i in range(len(kernel_sizes)):
            self.conv_from_bottleneck.append(InvConv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_sizes[i], 
                stride=1, 
                padding=kernel_sizes[i]//2, 
                init_type = init_type
            ))

    def forward(self, X):
        Z, coef_matrix = [],[]
        for i in range(len(self.conv_from_bottleneck)):
            out = self.conv_from_bottleneck[i](X)
            Z.append(out[0])
            coef_matrix.append(out[1])
        Z = torch.cat(Z, dim=1)  
        coef_matrix = torch.cat(coef_matrix, dim=1)
        return Z, coef_matrix
    

def set_kernel_size(kernel_sizes): 
        max_size = np.max(kernel_sizes)
        for size in kernel_sizes: 
            if max_size%size != 0: 
                raise ValueError(f"{max_size} is not a multiple of {size}.")
        return np.sort(kernel_sizes)[::-1]

class MultiScaleInvConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[128,64,32,16], bias= (True,True,True), init_type="gaussian"):
        # kernel size must be an interger in the poser of two or a list
        super(MultiScaleInvConv, self).__init__()

        self.kernel_sizes = set_kernel_size(kernel_sizes)

        self.conv_scales = nn.ModuleList()  
        for size in self.kernel_sizes:
            self.conv_scales.append(InvConv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=int(size), 
                stride=1, 
                padding=int(size//2), 
                init_type = init_type,
                bias=bias
            ))            

    def forward(self, X):
        Z, coef_matrix = [], []
        for conv in self.conv_scales:
            out = conv(X)
            Z.append(out[0])
            coef_matrix.append(out[1])
    
        Z = torch.cat(Z, dim=1)  
        coef_matrix = torch.cat(coef_matrix, dim=1)
        return Z, coef_matrix
    

class SparseMultiScaleConv(nn.Module): 

    def __init__(self,in_channels, out_channels,kernel_sizes=[128,64,32,16], bias = True, init_type="gaussian"):
        super(SparseMultiScaleConv,self).__init__()
        self.kernel_sizes = set_kernel_size(kernel_sizes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_bias = bias

        self.conv_scales = nn.ModuleList()
        max_size = self.kernel_sizes[0]
        for i,size in enumerate(self.kernel_sizes): 
            k_size = max_size //size
        
            self.conv_scales.append(Conv1d(
                in_channels=in_channels[i],
                out_channels=out_channels,
                kernel_size=int(k_size),
                padding= 0, #k_size//2,
                stride=1,
                dilation= int(size) if k_size>1 else 1,
                init_type= init_type,
                bias=False
            ))

        self.bias = nn.Parameter(torch.randn(1,self.out_channels,1)) if bias else None 

    def forward(self, X):
        input_per_kernel_size = torch.split(X,self.in_channels,dim=1)
        Z = []

        for input, conv, k_size in zip(input_per_kernel_size,self.conv_scales,self.kernel_sizes):
            result = conv(input[:,:,:-int(k_size)+1])
            Z.append(result.view((1,*result.shape)))

        Z = torch.concat(Z)
        Z = torch.sum(Z, dim=0)  
        if self.with_bias:
            return Z + self.bias
        else:
            return Z