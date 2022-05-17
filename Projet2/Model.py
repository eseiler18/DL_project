import pickle
import torch
import os
from tqdm import trange
from Modules import Sequential, Conv2d, ReLU, NearestUpsampling, Sigmoid
from Optimizers import SGD
from Losses import MSE
from path import OUT_DIR
from metrics import calculate_psnr
from utils import show

class Model():
    def __init__(self, input_channels=3, output_channels=3) -> None:

        self.model = Sequential(Conv2d(input_channels, 48, kernel_size=3,
                                       stride=2, padding=1),
                                ReLU(),
                                Conv2d(48, 48, kernel_size=3, stride=2,
                                       padding=1),
                                ReLU(),
                                NearestUpsampling(2),
                                Conv2d(48, 48, 3, padding=1),
                                ReLU(),
                                NearestUpsampling(2),
                                Conv2d(48, output_channels, 3, padding=1),
                                Sigmoid())

        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=0.1)

    def forward(self, img):
        return self.model.forward(img)

    def train(self, train_input, train_target, nb_epochs, batch_size=1,
              save=False) -> None:
        for e in trange(1, nb_epochs + 1, desc='Training', unit='epoch'):
            epoch_loss = 0
            for b in range(0, train_input.size(0), batch_size):
                self.model.zero_grad()
                output = self.model.forward(train_input[b:b+batch_size])
                loss = self.criterion.forward(output,
                                              train_target[b:b+batch_size])
                epoch_loss += loss
                output_grad = self.criterion.backward()
                self.model.backward(output_grad)
                self.optimizer.step(self.model)

            if e % 10 == 0:
                print(f"Epoch {e+1} \tLoss {epoch_loss}")
        if save:
            self.save_model()

    def predict(self, test_input) -> torch.Tensor:
        out = self.model.forward(test_input)
        return out

    def validation(self, noisy, clean):
        valid_loss = []
        valid_pnsr = []
        denoised = []
        for x_noised, x_clean in zip(noisy, clean):

            x_noised = x_noised.unsqueeze(0)
            x_clean = x_clean.unsqueeze(0)
            x_denoised = self.model.forward(x_noised)
            denoised.append(x_denoised[0,:,:,:])
            loss = self.criterion.forward(x_denoised, x_clean)
            psnr = calculate_psnr(x_denoised, x_clean)

            valid_loss.append(loss.item())
            valid_pnsr.append(psnr.item())

        avg_loss = sum(valid_loss)/len(valid_loss)
        avg_pnsr = sum(valid_pnsr)/len(valid_pnsr)
        print('for our model :')
        print(f'Hit psnr = {avg_pnsr} dB')
        print(f'Hit loss = {avg_loss}')
        
        val_img_ind = [2, 5, 6]
        for img in val_img_ind:
            show([noisy[img], denoised[img], clean[img]])

    def load_pretrained_model(self, name="our_model.pickle"):
        with open(os.path.join(OUT_DIR, name), 'rb') as file:
            param = pickle.load(file)
            a = 0
            for i, m in enumerate(self.model.modules):
                if isinstance(m, Conv2d):
                    m.kernel = param[a][0]
                    m.gradkernel = param[a][1]
                    a = a + 1
                    m.biais = param[a][0]
                    m.gradbiais = param[a][1]
                    a = a + 1

    def save_model(self) -> None:
        with open(os.path.join(OUT_DIR, "our_model.pickle"), 'wb') as f:
            param = self.model.param()
            pickle.dump(param, f)
