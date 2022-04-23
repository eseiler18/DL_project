class Optimizer(object):
    def step(self):
        raise NotImplementedError

        
class SGD(Optimizer):
    '''
    Stochastic gradient descent.
    '''
    def __init__(self, params, lr):
        super(Optimizer).__init__()
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = params
        self.lr = lr
    
    def step(self):
        '''
        Optimization step:
        '''
        for param in self.params:
            weight, grad = param
            if (weight is None) or (grad is None):
                continue
            else:
                weight.add_(-self.lr * grad) 
                