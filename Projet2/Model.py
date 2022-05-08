from Modules import *
from Optimizers import SGD
from Losses import MSE

class Model() :
    def __init__(self,input_channels=3,output_channels=3) -> None:
        
        self.model =  Sequential(Conv2d(input_channels,48,kernel_size=3,stride=2,padding=1),
                            ReLU(),
                            Conv2d(48,48,kernel_size=3,stride=2,padding=1),
                            ReLU(),
                            NearestUpsampling(2),
                            Conv2d(48, 48, 3,padding=1),
                            ReLU(),
                            NearestUpsampling(2),
                            Conv2d(48, output_channels, 3,padding=1),
                            Sigmoid())
        
        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=0.1)


    def forward(self,img):
        return self.model.forward(img)

    def load_pretrained_model(self) -> None:
        self.model = torch.load('bestmodel.pth')


    def train(self,train_input,train_target,nb_epochs, batch_size=1, save=False) -> None:
        batch_size = 1 # can add this to the parameters'
        
        for e in range(nb_epochs):
            epoch_loss = 0

            for b in range(0, train_input.size(0), batch_size):
                self.model.zero_grad()
                output = self.model.forward(train_input[b:b+batch_size])
                loss = self.criterion.forward(output, train_target[b:b + batch_size])
                epoch_loss += loss

                output_grad = self.criterion.backward()
                self.model.backward(output_grad)
                self.optimizer.step()

            if e % 10 == 0:
                print(f"Epoch {e+1} \tLoss {epoch_loss}")

        if save:
            torch.save(self.model, 'bestmodel.pth')
            
    def save_model(self ) -> None:
        torch.save(self.model, 'bestmodel.pth')
    
    def predict(self, test_input) -> torch.Tensor:
        out = selt.model.forward(test_input)
        return out

