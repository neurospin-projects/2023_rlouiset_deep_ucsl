import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        hidden_dim = 256
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim * 1 * 1)),
            nn.Linear(hidden_dim, z_dim * 2),
        )

        self.locs = nn.Linear(hidden_dim, z_dim)
        self.scales = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.encoder(x)
        return hidden[:, :self.z_dim], hidden[:, self.z_dim:]


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        hidden_dim = 256
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, z_dim: int = 32):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.z_dim = z_dim

    def encode(self, x):
        mean, log_var = self.encoder(x)
        return mean, log_var

    def decode(self, reparameterized_latent):
        return self.decoder(reparameterized_latent)

    def resample(self, z_mean, z_log_var):
        batch_size = z_mean.size()[0]

        z_std = z_log_var.mul(0.5).exp_()
        z_reparameterized = torch.randn((batch_size, self.z_dim))
        z_reparameterized = z_mean + z_std * z_reparameterized
        return z_reparameterized

    def forward(self, x):
        batch_size = x.size()[0]

        mean, log_var = self.encoder(x)

        std = log_var.mul(0.5).exp_()
        reparameterized_latent = torch.randn((batch_size, self.z_dim))
        reparameterized_latent = mean + std * reparameterized_latent

        hat_x = self.decoder(reparameterized_latent)
        return hat_x, mean, log_var
