import torch
import torch.nn as nn
import re
import sys
import importlib
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

class Model(nn.Module):
    def __init__(self):
        self.in_channels=3
        self.out_channel=64
        self.kernel_size=3
        self.n_stack=3
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channel, self.kernel_size, padding=1),
                                   nn.LeakyReLU(0.4))
        self.conv2 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, self.kernel_size, padding=1),
                                   nn.LeakyReLU(0.4))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(self.out_channel, self.out_channel, self.kernel_size,
                                     padding=1), nn.LeakyReLU(0.4))
        self.deconv2 = nn.ConvTranspose2d(self.out_channel, self.in_channels, self.kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.4)
        
        self.criterion=criterion = torch.nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-3)
        
    def forward(self, x):
        convs = []
        convs.append(self.conv1(x))
        for i in range(1, self.n_stack):
            convs.append(self.conv2(convs[i-1]))
        y = self.deconv1(convs[self.n_stack-1]) + convs[self.n_stack-2]
        for i in range(1, self.n_stack-1):
            y = self.deconv1(y)
            if i % 2 == 0:
                y = y.clone() + convs[self.n_stack-2-i]
                y = self.leakyrelu(y)
        y = self.deconv2(y)
        return y
    
    def load_pretrained_model(self):
        path=Path(__file__).parent / "bestmodel.pth"
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        
    def train(self, train_input,train_target, num_epochs=1):
        train_input=train_input.float()
        train_target=train_target.float()
        batch_size=100
        
        for epoch in range(num_epochs):
             with tqdm(zip(train_input.split(batch_size),train_target.split(batch_size)), total=train_input.shape[0]/batch_size,
                         unit='batch') as t:
                for x_train,y_train in t:
                    output=self.forward(x_train)
                    loss=self.criterion(output,y_train)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    
    def predict(self,test_input):
        test_input=test_input.float()
        out=self.forward(test_input)
        out=torch.clamp(out,0,255) # because of test_forward_dummy_input output should be in [0,255]
        return out 
        