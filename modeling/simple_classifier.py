import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs, embSize=256):
        super(MLP, self).__init__()
        self.embSize = embSize
        self.dim = n_inputs
        # self.layer = nn.Sequential(
        #     nn.Linear(n_inputs, embSize),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(embSize, n_outputs)
        # )
        self.lm1 = nn.Sequential(
            nn.Linear(n_inputs, embSize),
            nn.LeakyReLU(0.2, True)
        )
        self.lm2 = nn.Linear(embSize, n_outputs)
        self.penultimate_layer = None

    # forward propagate input
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = self.lm1(x)
        self.penultimate_layer = emb
        out = self.lm2(emb)
        return out

    def get_penultimate_dim(self):
        return self.embSize

class Network(nn.Module):
    def __init__(self, input_dim,  output_dim, hidden_layers=256):
        super(Network, self).__init__()
        self.embSize = hidden_layers
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x
    def get_penultimate_dim(self):
        return self.embSize