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

class Convolution(Module):
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
        if type(stride) == int:
            self.stride = (stride,stride)
        else:
            self.stride = stride
                
        self.padding = padding
        self.dilatation = dilatation
        self.initOption = initOption
        
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
        for channel in range(self.out_channels):
            #image = self.padding()
            shape = image.shape
            rv = 0
            cimg = []
            for r in range(self.kernel_size[0], shape[0]+1, self.stride[0]):
                cv = 0
                for c in range(self.kernel_size[1], shape[1]+1, self.stride[1]):
                    chunk = image[rv:r, cv:c]
                    soma = chunk * self.kernel[channel, :, :, :]
                    summa = soma.sum() + self.biases[channel]
                    cimg.append(summa)
                    cv+=stride[1]
                rv+=stride[0]
            cimg = np.array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
            
    def backward(self):
        

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
