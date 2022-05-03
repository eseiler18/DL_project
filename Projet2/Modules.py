import torch
import math
import numpy as np
from torch import cat
from torch.nn.functional import fold, unfold
##
from typing import Tuple
from torch import empty , cat , arange
from math import sqrt
#----------------------------------SuperClass Module----------------------------------------------------------

class Module(object):
    def forward(self, *x):
        raise NotImplementedError
    def backward(self, *upstream_derivative):
        raise NotImplementedError
    def param(self):
        return []
    def zero_grad(self):
        return
#----------------------------------Convolution Layer----------------------------------------------------------
class Conv2d(Module):

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    stride: Tuple[int, int]


    input: torch.Tensor
    input_unf: torch.Tensor

    weight: torch.Tensor
    weight_grad: torch.Tensor

    bias: torch.Tensor
    bias_grad: torch.Tensor
    bias_flag: bool

    output_size: Tuple[int, int, int, int]


    def compute_output_size(self, n, out_channels, win, hin, kernel_size, padding, dilation, stride):

        hout = int((hin + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1)/stride[0] + 1)
        wout = int((win + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1)/stride[1] + 1)

        return (n, out_channels, hout, wout)

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, stride=1, bias=True):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        weight_size = (out_channels, in_channels, kernel_size[0], kernel_size[1]) #from conv2d doc

        # initialize weight like in pytorch
        k = 1/(in_channels*kernel_size[0]*kernel_size[1])
        self.weight = empty(weight_size).uniform_(-sqrt(k), sqrt(k))
        self.weight_grad = empty(weight_size).fill_(0.0)

        # initialize bias to zero
        self.bias = empty(out_channels)
        self.bias_grad = empty(out_channels).fill_(0.0)
        self.bias_flag = bias

    def forward(self, input):
        self.input_size = input.size()
        print(self.input_size)
        # unfold input
        self.input_unf = unfold(input, kernel_size=self.kernel_size, padding=self.padding,
                                stride=self.stride, dilation=self.dilation)
        print('Weight PH')
        print(self.weight)
        
        wxb = self.weight.view(self.out_channels, -1) @ self.input_unf
        
        print('convolved PH')
        print(wxb)
        
        if self.bias_flag:
            wxb.add_(self.bias.view(1, -1, 1))

        self.output_size = self.compute_output_size(
            input.size(0), self.out_channels, input.size(2), input.size(3), self.kernel_size, self.padding,
            self.dilation, self.stride)

        self.output_unf_size = wxb.size()
        
        return wxb.view(self.output_size)

    def backward(self, gradwrtoutput):
        # revert reshaping at the end of forward
        gradwrtoutput_unf = gradwrtoutput.view(self.output_unf_size)

        # compute grad w.r.t bias: sum over batch (dim=0) and over the blocks (dim=2)
        bias_grad_unf = gradwrtoutput_unf.sum(dim=(0, 2))
        self.bias_grad.add_(bias_grad_unf.view(self.out_channels))

        # compute grad w.r.t. weight: like in the linear case for the unfolded tensor (modulo transpose)
        # and then reshaping to folded shape
        weight_grad_unf = self.input_unf @ gradwrtoutput_unf.transpose(1, 2)
        weight_grad_fold = weight_grad_unf.view(-1, self.out_channels, self.in_channels,
                                                self.kernel_size[0], self.kernel_size[1])
        # sum gradients over batch
        self.weight_grad.add_(weight_grad_fold.sum(dim=0))

        # compute grad w.r.t. input: like in the linear  case for the unfolded tensor
        # and then folding to the input size -> overlapping blocks are added which accumulates the gradients
        gradwrtinput_unf = self.weight.view(self.out_channels, -1).t() @ gradwrtoutput_unf
        gradwrtinput = fold(gradwrtinput_unf, output_size=(self.input_size[2], self.input_size[3]),
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)

        return gradwrtinput

    def param(self):
        return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]


class Conv2dbis(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, biais = True, initOption='Normal'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Two ways of defining the kernel size
        if type(kernel_size) == int:
            # 1st option : provide only one int, i.e. the kernel has same height and width
            self.kernel_size = (kernel_size,kernel_size)
        else:
            # provide directly an array, with possible different height and width
            self.kernel_size = kernel_size
        
        # Two ways of defining the stride
        if type(stride)== int :
            self.stride = (stride,stride)
        else:
            self.stride = stride
                
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.initOption = initOption
        #self.p = 1 if padding != None else 0
        #self.kernel = torch.empty(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
        #self.gradkernel = torch.empty(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
        
        weight_size = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        k = 1/(in_channels*kernel_size[0]*kernel_size[1])
        self.kernel = empty(weight_size).uniform_(-sqrt(k), sqrt(k))
        self.gradkernel = empty(weight_size).fill_(0.0)
        
        if biais:
            self.biais = torch.empty(self.out_channels)
            self.gradbiais = torch.empty(self.out_channels)
        else:
            self.biais = None
            self.gradbiais = None
            
        self.initOption = initOption
        self.initParameters()
        
    def initParameters(self):
        '''
        Different methods for parameter initialization.
        '''
        if self.initOption == 'Normal':
            self.kernel.normal_()
            if self.biais is not None:
                self.biais.normal_()
        if self.initOption == 'Zero': 
            self.kernel.zero_()
            if self.biais is not None:
                self.biais.zero_()
        self.gradkernel.zero_()
        if self.gradbiais is not None:
            self.gradbiais.zero_()
              
    # Sources Backprop : https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
    # https://github.com/zishansami102/CNN-from-Scratch/blob/master/MNIST/convnet.py
    # https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b
    # https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
    # https://q-viper.github.io/2020/06/05/convolutional-neural-networks-from-scratch-on-python/#312-conv2d-layer
    
    def forward(self,image):
        # image.size = NbImages,NbChannels,NbRows,NbCols
        self.input = image
        shape = image.size()
        print(shape)
        self.input_size = shape
        
        hout = int((shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
        wout = int((shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1] + 1)
        
        output_shape = (shape[0], self.out_channels, hout, wout)
        
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        
        print('Weight MY')
        print(self.kernel)
        
        convolved = self.kernel.view(self.out_channels, -1)@unfolded
        print('convolved MY')
        print(convolved)
        
        #out = convolved.reshape(shape[0], self.out_channels, NbOutRows, NbOutCols)
        out = convolved.view(output_shape)
        
        if self.biais is not None:
            out += self.biais.reshape(1, -1, 1, 1)
            
        self.convolved_shape = convolved.size()
        self.unfolded = unfolded
        self.kernel_unfolded = self.kernel.reshape(self.out_channels, -1).t()
        
    # Sources : https://github.com/coolgpu/Demo_Conv2d_forward_and_backward/blob/master/my_conv2d_v1.py
    # https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html#_Custom_implementation1
    
        return out
    
    def backward(self,upstream_derivative):
        
        shape = self.input.size()
        
        # grad wrt biais: sum over batch (dim=0) and over the blocks (dim=2,3)
        grad_biais = upstream_derivative.sum(dim=[0, 2, 3])

        self.gradbiais = grad_biais
        #grad wrt kernel 
        
        # reshape upstream derivative to match the shape before the folding operation in the forward pass
        upstream_derivative_unfolded = upstream_derivative.reshape(self.convolved_shape)
               
        grad_kernel_unfolded = self.unfolded.matmul(upstream_derivative_unfolded.transpose(1, 2))
        grad_kernel = grad_kernel_unfolded.view(-1,self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])  
        
        
        self.gradkernel = grad_kernel.sum(dim=0)
        
              
        # grad wrt input
        
        grad_input_folded = self.kernel.view(self.out_channels, -1).t()@upstream_derivative_unfolded

        grad_input = torch.nn.functional.fold(grad_input_folded,(self.input_size[2], self.input_size[3]), (self.kernel_size[0], self.kernel_size[1]),padding=self.padding, stride=self.stride, dilation=self.dilation)
        

        return grad_input
    
    def backward2(self, gradwrtoutput):
        self.input_size = self.input.size()
        # revert reshaping at the end of forward
        gradwrtoutput_unf = gradwrtoutput.view(self.convolved_shape)

        # compute grad w.r.t bias: sum over batch (dim=0) and over the blocks (dim=2)
        bias_grad_unf = gradwrtoutput_unf.sum(dim=(0, 2))
        self.gradbiais.add_(bias_grad_unf.view(self.out_channels))

        # compute grad w.r.t. weight: like in the linear case for the unfolded tensor (modulo transpose)
        # and then reshaping to folded shape
        weight_grad_unf = self.unfolded @ gradwrtoutput_unf.transpose(1, 2)
        weight_grad_fold = weight_grad_unf.view(-1, self.out_channels, self.in_channels,
                                                self.kernel_size[0], self.kernel_size[1])
        # sum gradients over batch
        self.gradkernel.add_(weight_grad_fold.sum(dim=0))

        # compute grad w.r.t. input: like in the linear  case for the unfolded tensor
        # and then folding to the input size -> overlapping blocks are added which accumulates the gradients
        gradwrtinput_unf = self.kernel.view(self.out_channels, -1).t() @ gradwrtoutput_unf
        gradwrtinput = fold(gradwrtinput_unf, output_size=(self.input_size[2], self.input_size[3]),
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)

        return gradwrtinput

    
    def param(self):
        '''
        Return parameters.
        '''
        if self.biais is not None:
            return [(self.kernel, self.gradkernel),
                    (self.biais, self.grad_biais)]
        else:
            return [(self.kernel, self.gradkernel)]
    
    def zero_grad(self):
        self.gradkernel.zero_()
        if self.biais is not None:
            self.grad_biais.zero_()
        

#----------------------------------Upsampling-----------------------------------------------------------------

class NearestUpsampling(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = (scale_factor,scale_factor)
        
    def forward(self,image):
        self.input = image
        shape = image.size()
        
        # First, upsamples the rows
        ups_rows=[]
        for i in range(shape[2]):
            # extract the row
            rows = image[:, :, i, :].view(shape[0], shape[1], 1, shape[3])
            # concatenate n times the extracted row, with n=scale_factor
            ups_rows.append(cat(self.scale_factor*[rows], dim=2))
        # concatenate the groups of repeted rows to construct the final upsampled image
        img_ups_rows = cat(ups_rows, dim=2)

        # Repeat for the colums, starting from the image with upsampled rows
        shape = img_ups_rows.size()
        print(shape)
        ups_cols=[]
        for i in range(shape[3]):
            # extract the row
            cols = img_ups_rows[:, :, :, i].view(shape[0], shape[1], shape[2], 1)
            # concatenate n times the extracted row, with n=scale_factor
            ups_cols.append(cat(self.scale_factor*[cols], dim=3))
        # concatenate the groups of repeted rows to construct the final upsampled image
        img_ups = cat(ups_cols, dim=3)
        
        return img_ups
    
    def backward(self,upstream_derivative):
        return torch.empty(self.input.size()).fill_(self.scale_factor**2)
    
    def param(self):
        return []
    

#----------------------------------Activation Functions-------------------------------------------------------
class ReLU(Module):
    def forward(self, x):
        self.x = x
        return torch.max(torch.zeros_like(x), x)
    
    def backward(self, upstream_derivative):
        return torch.clamp(self.x.sign(), 0.0, 1.0).mul(upstream_derivative)
    
    def param(self):
        return []

class Tanh(Module):
    def forward(self,x):
        self.x = x
        return torch.tanh(x)
    
    def backward(self,upstream_derivative):
        return 4 * ((self.x.exp() + self.x.mul(-1).exp()).pow(-2)) * upstream_derivative
    
    def param(self):
        return []
    
class Sigmoid(Module):
    def forward(self,x):
        self.x = x
        return torch.div(1, (1+ torch.exp(-self.x)))
    def backward(self,upstream_derivative):
        sigma = torch.div(1, (1+ torch.exp(-self.x)))
        return upstream_derivative * (sigma - sigma**2)
    def param(self):
        return []
    
class LeakyReLu(Module): #REVOIRRRR
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return x.clamp(min=0.0) + self.alpha*x.clamp(max=0.0)

    def backward(self, grdwrtoutput):
        drelu = torch.ones(self.s.size())
        drelu[self.s < 0] = self.alpha
        return grdwrtoutput * drelu

    def param(self):
        return []

#----------------------------------Sequential Container-------------------------------------------------------

class Sequential(Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        self.x = x
        for m in self.modules:
            x = m.forward(x)
        return x

    def backward(self, gradout):
        reversed_modules = self.modules[::-1]
        out = gradout
        for m in reversed_modules:
            out = m.backward(out)
        return out

    def param(self):
        param = []
        for m in self.modules:
            param.extend(m.param())
        return param

    def zero_grad(self):
        for m in self.modules:
            m.zero_grad()
            
    def add_module(self, ind, module):
        self.ind = module
        self.modules.append(self.ind)
        return module
