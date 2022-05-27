import pickle
import torch
import os
import importlib
from pathlib import Path
from tqdm import trange

from .others.Modules import Sequential, Conv2d, ReLU, NearestUpsampling, Sigmoid
from .others.Optimizers import SGD
from .others.Losses import MSE
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

class Model():
    def __init__(self):
        input_channels=3 
        output_channels=3
        kernel_size=3
        nbchannels1=256 
        nbchannels2=32
        nbchannels3=256

        self.model = Sequential(Conv2d(input_channels, nbchannels1,
                                       kernel_size=kernel_size,
                                       stride=2, padding=1),
                                ReLU(),
                                Conv2d(nbchannels1, nbchannels2,
                                       kernel_size=kernel_size, stride=2,
                                       padding=1),
                                ReLU(),
                                NearestUpsampling(2),
                                Conv2d(nbchannels2, nbchannels3,
                                       kernel_size=kernel_size, padding=1),
                                ReLU(),
                                NearestUpsampling(2),
                                Conv2d(nbchannels3, output_channels,
                                       kernel_size=kernel_size, padding=1),
                                Sigmoid())

        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=0.1)

        nb1 = kernel_size*kernel_size*input_channels*nbchannels1 + nbchannels1
        nb2 = kernel_size*kernel_size*nbchannels1*nbchannels2 + nbchannels2
        nb3 = kernel_size*kernel_size*nbchannels2*nbchannels3 + nbchannels3
        nb4 = kernel_size*kernel_size*nbchannels3*output_channels + output_channels

        self.nb_params = nb1+nb2+nb3+nb4

    def load_pretrained_model(self):
        
        with open(Path(__file__).parent / "bestmodel.pth", 'rb') as file:
            param = pickle.load(file)
           
            a = 0
            for i, m in enumerate(self.model.modules):
                if isinstance(m, Conv2d):
                    m.weight = param[a][0]
                    m.gradkernel = param[a][1]
                    a += 1
                    m.bias = param[a][0]
                    m.gradbias = param[a][1]
                    a += 1
    def train(self, train_input, train_target, num_epochs=1):
        train_input=train_input.float()/255.0 # to fit the test script input range
        train_target=train_target.float()/255.0
        for epoch in trange(1, num_epochs + 1, desc='Training', unit='epoch'):
            
            batch_size=100
            
            with tqdm(zip(train_input.split(batch_size),train_target.split(batch_size)), total=train_input.shape[0]/batch_size,
                         unit='batch') as t:
                for x_train,y_train in t:
                    self.model.zero_grad()
                    output = self.model.forward(x_train)
                    self.criterion.forward(output, y_train)
                    output_grad = self.criterion.backward()
                    self.model.backward(output_grad)
                    self.optimizer.step(self.model)

    def predict(self, test_input):
        test_input=test_input.float()/255.0 # to fit the test script input range
        return self.model.forward(test_input)*255.0

    def forward(self, img):
        fw=self.model.forward(img)*255
        print(fw.min(),fw.max())
        return self.model.forward(img)*255.0
