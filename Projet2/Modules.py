import torch
from torch import empty, cat
from math import sqrt

# ------------------------------SuperClass Module------------------------------


class Module(object):
    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *upstream_derivative):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        return
# -------------------------- ---Convolution Layer------------------------------


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, biais=True, initOption='Normal'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Two ways of defining the kernel size
        if type(kernel_size) == int:
            # 1st option : provide only one int, i.e. the kernel has same
            # height and width
            self.kernel_size = (kernel_size, kernel_size)
        else:
            # provide directly an array, with possible different height and
            # width
            self.kernel_size = kernel_size

        # Two ways of defining the stride
        if type(stride) == int:
            self.stride = (stride, stride)
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
        # self.p = 1 if padding != None else 0
        # self.kernel = torch.empty(self.out_channels,self.in_channels,
        # self.kernel_size[0],self.kernel_size[1])
        # self.gradkernel = torch.empty(self.out_channels,self.in_channels,
        # self.kernel_size[0],self.kernel_size[1])

        weight_size = (out_channels, in_channels, self.kernel_size[0],
                       self.kernel_size[1])
        k = 1/(in_channels * self.kernel_size[0] * self.kernel_size[1])
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

    # Sources Backprop :
    # https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
    # https://github.com/zishansami102/CNN-from-Scratch/blob/master/MNIST/convnet.py
    # https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b
    # https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
    # https://q-viper.github.io/2020/06/05/convolutional-neural-networks-from-scratch-on-python/#312-conv2d-layer

    def forward(self, image):
        # image.size = NbImages,NbChannels,NbRows,NbCols
        self.input = image
        shape = image.size()

        self.input_size = shape

        hout = int((shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
        wout = int((shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1] + 1)

        output_shape = (shape[0], self.out_channels, hout, wout)

        unfolded = torch.nn.functional.unfold(image,
                                              kernel_size=self.kernel_size,
                                              padding=self.padding,
                                              stride=self.stride,
                                              dilation=self.dilation)

        convolved = self.kernel.view(self.out_channels, -1)@unfolded
        # out = convolved.reshape(shape[0], self.out_channels, NbOutRows,
        #                         NbOutCols)
        out = convolved.view(output_shape)

        if self.biais is not None:
            out += self.biais.reshape(1, -1, 1, 1)

        self.convolved_shape = convolved.size()
        self.unfolded = unfolded
        self.kernel_unfolded = self.kernel.reshape(self.out_channels, -1).t()

    # Sources : https://github.com/coolgpu/Demo_Conv2d_forward_and_backward/blob/master/my_conv2d_v1.py
    # https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html#_Custom_implementation1

        return out

    def backward(self, upstream_derivative):

        # shape = self.input.size()

        # grad wrt biais: sum over batch (dim=0) and over the blocks (dim=2,3)
        grad_biais = upstream_derivative.sum(dim=[0, 2, 3])

        self.gradbiais = grad_biais
        # grad wrt kernel

        # reshape upstream derivative to match the shape before the folding
        # operation in the forward pass
        upstream_derivative_unfolded = upstream_derivative.reshape(self.convolved_shape)

        grad_kernel_unfolded = self.unfolded.matmul(upstream_derivative_unfolded.transpose(1, 2))
        grad_kernel = grad_kernel_unfolded.view(-1, self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        self.gradkernel = grad_kernel.sum(dim=0)

        # grad wrt input

        grad_input_folded = self.kernel.view(self.out_channels, -1).t()@upstream_derivative_unfolded

        grad_input = torch.nn.functional.fold(grad_input_folded, (self.input_size[2], self.input_size[3]), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride=self.stride, dilation=self.dilation)

        return grad_input

    def param(self):
        '''
        Return parameters.
        '''
        if self.biais is not None:
            return [(self.kernel, self.gradkernel),
                    (self.biais, self.gradbiais)]
        else:
            return [(self.kernel, self.gradkernel)]

    def zero_grad(self):
        self.gradkernel.zero_()
        if self.biais is not None:
            self.gradbiais.zero_()


# ----------------------------------Upsampling---------------------------------

class NearestUpsampling(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = (scale_factor, scale_factor)

    def forward(self, image):
        self.input = image
        shape = image.size()

        # First, upsamples the rows
        ups_rows = []
        for i in range(shape[2]):
            # extract the row
            rows = image[:, :, i, :].view(shape[0], shape[1], 1, shape[3])
            # concatenate n times the extracted row, with n=scale_factor
            ups_rows.append(cat(self.scale_factor*[rows], dim=2))
        # concatenate the groups of repeted rows to construct the final
        # upsampled image
        img_ups_rows = cat(ups_rows, dim=2)

        # Repeat for the colums, starting from the image with upsampled rows
        shape = img_ups_rows.size()
        ups_cols = []
        for i in range(shape[3]):
            # extract the row
            cols = img_ups_rows[:, :, :, i].view(shape[0], shape[1], shape[2],
                                                 1)
            # concatenate n times the extracted row, with n=scale_factor
            ups_cols.append(cat(self.scale_factor*[cols], dim=3))
        # concatenate the groups of repeted rows to construct the final
        # upsampled image
        img_ups = cat(ups_cols, dim=3)
        return img_ups

    def backward(self, upstream_derivative):
        # corriger ça
        shape = self.input.size()
        self.derivative = torch.empty(shape).fill_(0)

        for i in range(shape[2]):
            for j in range(shape[3]):
                self.derivative[:, :, i, j] = upstream_derivative[:, :, i*self.kernel_size[0]:(i+1)*self.kernel_size[0], j*self.kernel_size[1]:(j+1)*self.kernel_size[1]].sum(dim=(2, 3))
        return self.derivative

    def param(self):
        return []


# ----------------------------------Activation Functions-----------------------
class ReLU(Module):
    def forward(self, x):
        self.x = x
        return torch.max(torch.zeros_like(x), x)

    def backward(self, upstream_derivative):
        return torch.clamp(self.x.sign(), 0.0, 1.0).mul(upstream_derivative)

    def param(self):
        return []


class Tanh(Module):
    def forward(self, x):
        self.x = x
        return torch.tanh(x)

    def backward(self, upstream_derivative):
        return 4 * ((self.x.exp() + self.x.mul(-1).exp()).pow(-2)) * upstream_derivative

    def param(self):
        return []


class Sigmoid(Module):
    def forward(self, x):
        self.x = x
        return torch.div(1, (1 + torch.exp(-self.x)))

    def backward(self, upstream_derivative):
        sigma = torch.div(1, (1 + torch.exp(-self.x)))
        return upstream_derivative * (sigma - sigma**2)

    def param(self):
        return []


class LeakyReLu(Module):
    # REVOIRRRR
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

# ----------------------------------Sequential Container-----------------------


class Sequential(Module):
    def __init__(self, *args):

        super().__init__()
        # for  module in modules: #check ces deux lignes
        #   module.__init__()
        self.modules = []
        for module in args:
            self.modules.append(module)

    def forward(self, x):
        self.x = x
        for m in self.modules:
            x = m.forward(x)
            # print(x.shape)
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
