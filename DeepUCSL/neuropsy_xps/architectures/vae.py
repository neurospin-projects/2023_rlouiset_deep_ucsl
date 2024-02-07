import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, channel=512, z_dim=256):
        super(Encoder, self).__init__()

        self.channel = channel

        self.conv1 = nn.Conv3d(1, channel // 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel // 16, channel // 8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel // 8)
        self.conv3 = nn.Conv3d(channel // 8, channel // 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel // 4)
        self.conv4 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel // 2)
        self.conv5 = nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(channel)

        self.mean = nn.Sequential(
            nn.Linear(32768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, z_dim))
        self.logvar = nn.Sequential(
            nn.Linear(32768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, z_dim))

    def forward(self, x, _return_activations=False):
        batch_size = x.size()[0]

        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)

        mean = self.mean(h5.view(batch_size, -1))
        logvar = self.logvar(h5.view(batch_size, -1))

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, channel: int = 512):
        super(Decoder, self).__init__()
        _c = channel

        self.fc = nn.Linear(256, 512 * 4 * 4 * 4)
        self.bn1 = nn.BatchNorm3d(_c)

        self.tp_conv2 = nn.ConvTranspose3d(channel, channel // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channel // 2)

        self.tp_conv3 = nn.ConvTranspose3d(channel // 2, channel // 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channel // 4)

        self.tp_conv4 = nn.ConvTranspose3d(channel // 4, channel // 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(channel // 8)

        self.tp_conv5 = nn.ConvTranspose3d(channel // 8, channel // 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(channel // 16)

        self.tp_conv6 = nn.ConvTranspose3d(channel // 16, 1, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, latent):
        latent = latent.view(-1, 256)
        h = self.fc(latent)
        h = h.view(-1, 512, 4, 4, 4)
        h = F.relu(self.bn1(h))
        h = F.relu(self.bn2(self.tp_conv2(h)))
        h = F.relu(self.bn3(self.tp_conv3(h)))
        h = F.relu(self.bn4(self.tp_conv4(h)))
        h = F.relu(self.bn5(self.tp_conv5(h)))
        h = self.tp_conv6(h)
        # h = torch.sigmoid(h)
        return h

class BrainVAE(nn.Module):
    def __init__(self, z_dim: int = 256, encoder_channel: int = 512, decoder_channel: int = 512):
        super(BrainVAE, self).__init__()
        self.encoder = Encoder(encoder_channel)
        self.decoder = Decoder(decoder_channel)
        self.z_dim = z_dim

    def forward(self, x):
        device = x.get_device()
        batch_size = x.size()[0]

        mean, log_var = self.encoder(x)

        std = log_var.mul(0.5).exp_()
        reparameterized_latent = torch.randn((batch_size, self.z_dim), device=device)
        reparameterized_latent = mean + std * reparameterized_latent

        hat_x = self.decoder(reparameterized_latent)
        return hat_x, mean, log_var
