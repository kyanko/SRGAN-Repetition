import torch
import torch.nn as nn
import numpy as np


class SRGANResBlock(nn.Module):
    def __init__(self):
        super(SRGANResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return x + self.block(x)


class SRGANGenerator(nn.Module):
    def __init__(self):
        super(SRGANGenerator, self).__init__()
        self.PreLayer = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )
        self.ResidualLayer = nn.Sequential(
            SRGANResBlock(),
            SRGANResBlock(),
            SRGANResBlock(),
            SRGANResBlock(),
            SRGANResBlock()
        )
        self.PostLayer = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.PostLayer2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(16, 3, 9, 1, 4)
        )

    def forward(self, x):
        Output = self.PreLayer(x)
        Output_copy = self.PreLayer(x)
        Output = self.ResidualLayer(Output)
        Output = self.PostLayer(Output) + Output_copy
        Output = self.PostLayer2(Output)
        return Output


class SRGANDiscriminator(nn.Module):
    def __init__(self):
        super(SRGANDiscriminator, self).__init__()
        self.ConvLayer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.DownLayer = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.DenseLayer = nn.Sequential(
            nn.Linear(512*8*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.ConvLayer(x)
        output = self.DownLayer(output)
        output = output.view(output.shape[0], -1)
        output = self.DenseLayer(output)
        return output


# GModel = SRGANGenerator()
# DModel = SRGANDiscriminator()
# x = np.random.normal(0, 1, (10, 3, 32, 32))
# x = torch.tensor(x, dtype=torch.float32)
# y = GModel(x)
# L = DModel(y)
# print(y.shape, L.shape)
