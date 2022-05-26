import pickle
import torch
import os
from tqdm import trange
from other.Modules import Sequential, Conv2d, ReLU, NearestUpsampling, Sigmoid
from other.Optimizers import SGD
from other.Losses import MSE
from other.path import OUT_DIR
from other.metrics import calculate_psnr
from other.utils import show


class Model():
    def __init__(self, input_channels=3, output_channels=3,
                 kernel_size=3, nbchannels1=256, nbchannels2=32,
                 nbchannels3=256) -> None:

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

    def load_pretrained_model(self, name="bestmodel_part2.pickle"):
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

    def train(self, train_input, train_target, nb_epochs=10, batch_size=1,
              save=False, savingname='our_model.pickle') -> None:
        for epoch in trange(1, nb_epochs + 1, desc='Training', unit='epoch'):
            epoch_loss = 0
            for batch in range(0, train_input.size(0), batch_size):
                self.model.zero_grad()
                output = self.model.forward(train_input[batch:batch+batch_size])
                loss = self.criterion.forward(output,
                                              train_target[batch:batch+batch_size])
                epoch_loss += loss
                output_grad = self.criterion.backward()
                self.model.backward(output_grad)
                self.optimizer.step(self.model)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1} \tLoss {epoch_loss}")
        if save:
            self.save_model(name=savingname)

    def predict(self, test_input) -> torch.Tensor:
        return self.model.forward(test_input)

    def validation(self, noisy, clean):
        valid_loss = []
        valid_pnsr = []
        denoised = []
        for x_noised, x_clean in zip(noisy, clean):

            x_noised = x_noised.unsqueeze(0)
            x_clean = x_clean.unsqueeze(0)
            x_denoised = self.model.forward(x_noised)
            denoised.append(x_denoised[0, :, :, :])
            loss = self.criterion.forward(x_denoised, x_clean)
            psnr = calculate_psnr(x_denoised, x_clean)

            valid_loss.append(loss.item())
            valid_pnsr.append(psnr.item())

        avg_loss = sum(valid_loss)/len(valid_loss)
        avg_pnsr = sum(valid_pnsr)/len(valid_pnsr)
        print('for our model :')
        print(f'Hit psnr = {avg_pnsr} dB')
        print(f'Hit loss = {avg_loss}')

        val_img_ind = [27]
        for img in val_img_ind:
            show([noisy[img], denoised[img], clean[img]])

        return avg_loss, avg_pnsr

    def forward(self, img):
        return self.model.forward(img)

    def save_model(self, name="bestmodel_part2.pickle") -> None:
        with open(os.path.join(OUT_DIR, name), 'wb') as f:
            param = self.model.param()
            pickle.dump(param, f)
