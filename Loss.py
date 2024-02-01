import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import numpy.fft as fft


class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        for para in vgg.parameters():
            para.requires_grad = True
        self.vgg = vgg.features[:19]
        self.vgg = self.vgg.to(device)

    def forward(self, x):
        return self.vgg(x)


class ContentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg19 = VGG(device)

    def forward(self, fake, real):
        feature_fake = self.vgg19(fake)/12.75
        feature_real = self.vgg19(real)/12.75
        loss = self.mse(feature_fake, feature_real)
        return loss


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(-torch.log(x))


class AllLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg_loss = ContentLoss(device)
        self.adversarial_loss = AdversarialLoss()

    def forward(self, fake, real, x):
        vgg_loss = self.vgg_loss(fake, real)
        adversarial_loss = self.adversarial_loss(x)
        return vgg_loss + 1e-3 * adversarial_loss


class HighFreqLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake, real):
        f1 = fft.fft2(np.transpose(fake.data, (0, 2, 3, 1)))
        f2 = fft.fft2(np.transpose(real.data, (0, 2, 3, 1)))
        f1 = fft.fftshift(f1)
        f2 = fft.fftshift(f2)
        f1[:, 40:56, 40:56, :] = 0
        f2[:, 40:56, 40:56, :] = 0
        f1 = fft.ifft2(fft.ifftshift(f1))
        f2 = fft.ifft2(fft.ifftshift(f2))
        f1 = np.abs(f1)
        f2 = np.abs(f2)
        f1 = np.transpose(f1, (0, 3, 1, 2))
        f2 = np.transpose(f2, (0, 3, 1, 2))
        f1 = torch.tensor(f1, dtype=torch.float32)
        f2 = torch.tensor(f2, dtype=torch.float32)

        return nn.MSELoss()(f1, f2)


# fake = torch.tensor(np.random.random((10, 3, 100, 100)), dtype=torch.float32)
# real = torch.tensor(np.random.random((10, 3, 100, 100)), dtype=torch.float32)
# gpu = torch.cuda.device("cuda")
# test = HighFreqLoss()
# loss = test(fake, real)
# print(loss.item())
