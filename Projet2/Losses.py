class Loss(object):
    def forward(self, output, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self,output,target):
        self.output = output
        selt.target = target.view(output.shape)
        loss = (self.target - self.output).pow(2)
        return torch.mean(loss,1).view(output.shape)
    
    def backward(self):
        back = torch.div(self.output - self.target, self.output.size(0)) * 2
        return back