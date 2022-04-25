import torch
import math
import numpy as np

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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, biais = True, initOption='Normal'):
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
        if type(stride) == int:
            self.stride = (stride,stride)
        else:
            self.stride = stride
                
        self.padding = padding
        self.dilatation = dilatation
        self.initOption = initOption
        self.p = 1 if padding != None else 0
        self.kernel = torch.empty(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
        self.gradkernel = torch.empty(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
        
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
    
    '''def padding(self, image):
        shape = image.shape
        if self.padding == "zero":
            zeros_h = np.zeros((shape[1], shape[2])).reshape(-1, shape[1], shape[2])
            zeros_v = np.zeros((shape[0]+2, shape[2])).reshape(shape[0]+2, -1, shape[2])
            padded_img = np.vstack((zeros_h, image, zeros_h)) # add rows
            padded_img = np.hstack((zeros_v, padded_img, zeros_v)) # add cols
            image = padded_img
        elif self.padding == "same":
            h1 = image[0].reshape(-1, shape[1], shape[2])
            h2 = image[-1].reshape(-1, shape[1], shape[2])
            padded_img = np.vstack((h1, image, h2)) # add rows
            v1 = padded_img[:, 0].reshape(padded_img.shape[0], -1, shape[2])
            v2 = padded_img[:, -1].reshape(padded_img.shape[0], -1, shape[2])
            padded_img = np.hstack((v1, padded_img, v2)) # add cols
            image = padded_img
        elif self.padding == None:
            pass
        return image''' 
    # padding impossible sans numpy non? ou alors est ce qu'il y a des equivalents à hstack/vstack?
    
        
    def forward(self,image):
        self.input = image
        shape = image.size()
        output_shape = (self.out_channels, int((shape[0] - kernel_size[0] + 2 * self.p) / stride[0]) + 1,
                            int((shape[1] - kernel_size[1] + 2 * self.p) / stride[1]) + 1)
        out = torch.empty(output_shape)
        for channel in range(self.out_channels):
            #image = self.padding()
            rv = 0
            cimg = []
            for r in range(self.kernel_size[1], shape[1]+1, self.stride[0]):
                cv = 0
                for c in range(self.kernel_size[2], shape[2]+1, self.stride[1]):
                    chunk = image[:,rv:r, cv:c]
                    soma = chunk * self.kernel[channel, :, :, :]
                    summa = soma.sum() + self.biases[channel]
                    cimg.append(summa)
                    cv+=self.stride[1]
                rv+=self.stride[0]
            cimg = np.array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
            out[channel, :, :] = cimg
        return out
            
    # Sources Backprop : https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
    # https://github.com/zishansami102/CNN-from-Scratch/blob/master/MNIST/convnet.py
    # https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b
    # https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
    # https://q-viper.github.io/2020/06/05/convolutional-neural-networks-from-scratch-on-python/#312-conv2d-layer
    
    def backward(self):
        

#----------------------------------Upsampling-----------------------------------------------------------------

class NearestUpsampling(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = (scale_factor,scale_factor)
        
    def forward(self,image):
        self.input = image
        shape = image.size()
        output_shape = (shape[0], shape[1] * self.kernel_size[0], shape[2] * self.kernel_size[1])
        out = torch.empty(output_shape)
        for channel in range(self.out_channels):
            #image = self.padding()
            rv = 0
            i = 0
            for r in range(self.kernel_size[0], shape[1]+1, self.kernel_size[0]):
                cv = 0
                j = 0
                for c in range(self.kernel_size[1], shape[2]+1, self.kernel_size[1]):
                    out[channel,rv:r, cv:c] = image[channel, i, j]
                    j+=1
                    cv+=self.kernel_size[1]
                i+=1
                rv+=self.kernel_size[0]
        return out  
    
    def backward(self,upstream_derivative):
        shape = self.input.size()
        self.derivative = np.zeros(shape) 
        
        for f in range(shape[2]):
            i = 0
            rv = 0
            for r in range(self.kernel_size[0], shape[0]+1,self.kernel_size[0]):
                cv = 0
                j = 0
                for c in range(self.kernel_size[1], shape[1]+1, self.kernel_size[1]):
                    dout = upstream_derivative[channel, rv:r, cv:c]
                    self.derivative[i, j] = dout.sum()
                    j+=1
                    cv+=cstep
                rv+=rstep
                i+=1
        return self.derivative
    
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
