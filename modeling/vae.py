import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
import torchvision.models as models

class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=16384, out_features=128)
        self.enc2 = nn.Linear(in_features=128, out_features=32)

        # decoder
        self.dec1 = nn.Linear(in_features=16, out_features=128)
        self.dec2 = nn.Linear(in_features=128, out_features=16384)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        #print('shape of x: ', x.shape)
        x = F.relu(self.enc1(x))
        #print('shape of x: ', x.shape)
        x = self.enc2(x).view(-1, 2, 16)
        #print('shape of x: ', x.shape)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        #print('shape of z: ', z.shape)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var
