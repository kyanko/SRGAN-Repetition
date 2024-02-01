import torch
import torchvision
import torch.utils.data as Data
import sys
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np

sys.path.append('../DataSet/STL10/')
sys.path.append('../ESRGAN/')
from CreateDataSet import CreateSTL10DataSet as CSTLF
from LoadDataSet import LoadHRLRImage
from CreateModel import SRGANDiscriminator, SRGANGenerator
from torchvision.utils import save_image
import cv2
import visdom
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from Loss import AllLoss, HighFreqLoss
from torch.autograd import Variable

train_vis = visdom.Visdom()

GPUDevice = torch.device("cuda")
CPUDevice = torch.device("cpu")
TrainY, TrainX = CSTLF(DataType='test', DataEnhance="resize", NewSize=(24, 24))
TestY0, TestX0 = CSTLF(DataType='train', DataEnhance="resize", NewSize=(24, 24))
TestImageN = 100
TestY = TestY0[:TestImageN]
TestX = TestX0[:TestImageN]

ValiY = TestY0[300:400]
ValiX = TestX0[300:400]
BatchSize = 64
TrainData = Data.TensorDataset(TrainY, TrainX)
train_loader = Data.DataLoader(
    dataset=TrainData,
    shuffle=True,
    batch_size=BatchSize,
    drop_last=True
)

GModel = SRGANGenerator()
DModel = SRGANDiscriminator()

GModel = GModel.to(GPUDevice)
DModel = DModel.to(GPUDevice)

MaxEpoch = 200
LR = 0.0001
GOptim = optim.Adam(GModel.parameters(), lr=LR, betas=(0.9, 0.999))
DOptim = optim.Adam(DModel.parameters(), lr=LR, betas=(0.9, 0.999))
MSEFunc = nn.MSELoss().to(GPUDevice)
NowStep = 0
train_vis.close(win='SRGAN DLoss')
train_vis.close(win='SRGAN GLoss')
train_vis.close(win='SRGAN SSIM')
train_vis.close(win='SRGAN PSNR')

GModelLossFunc = AllLoss(GPUDevice)
DLossFunc = nn.BCELoss()
Tensor = torch.cuda.FloatTensor
for epoch in range(MaxEpoch):
    for step, (RealImg, DownImg) in enumerate(train_loader):
        NowStep = NowStep + 1
        RealImg = RealImg.to(GPUDevice)
        DownImg = DownImg.to(GPUDevice)
        FakeImg = GModel(DownImg).to(GPUDevice)
        FakeLabel = DModel(FakeImg)
        RealLabel = DModel(RealImg)

        Label1 = Variable(Tensor(np.ones((BatchSize, 1, 1, 1))), requires_grad=True)
        Label0 = Variable(Tensor(np.zeros((BatchSize, 1, 1, 1))), requires_grad=True)

        DOptim.zero_grad()
        DLoss = torch.mean(FakeLabel) - torch.mean(RealLabel)
        DLoss = DLossFunc(FakeLabel, Label0) + DLossFunc(RealLabel, Label1)
        DLoss.backward(retain_graph=True)
        DOptim.step()

        GOptim.zero_grad()
        FakeLabel = DModel(FakeImg)
        RealLabel = DModel(RealImg)
        # GLoss = MSEFunc(FakeImg, RealImg)
        # GLoss = GModelLossFunc(FakeImg, RealImg, FakeLabel)

        GLoss = MSEFunc(FakeImg, RealImg) - torch.mean(FakeLabel)
        GLoss.backward()
        GOptim.step()

        if step % 20 == 0:
            print('Epoch=%d,NowStep=%d' % (epoch + 1, step + 1), 'GLoss=', GLoss.item(), 'DLoss=', DLoss.item())
            ValiFake = GModel(ValiX.to(GPUDevice))

            ValiFakeLabel = DModel(ValiFake)
            ValiLoss = GModelLossFunc(ValiFake, ValiY.to(GPUDevice), ValiFakeLabel)
            # ValiLoss = MSEFunc(ValiFake, ValiY.to(GPUDevice))
            ValiFake = ValiFake.to(CPUDevice)
            AllValSSIM = 0
            AllValPSNR = 0
            for imgi in range(100):
                vreal = np.transpose(ValiY.data[imgi].numpy(), (1, 2, 0))
                vfake = np.transpose(ValiFake.data[imgi].numpy(), (1, 2, 0))

                valissim = ssim(vfake, vreal, data_range=1, channel_axis=-1)
                valipsnr = psnr(vfake, vreal, data_range=-1)
                AllValSSIM = AllValSSIM + valissim / 100
                AllValPSNR = AllValPSNR + valipsnr / 100

            train_vis.line(
                X=[NowStep],
                Y=[AllValSSIM],
                opts=dict(title='SRGAN SSIM',
                          linecolor=np.array([[255, 0, 0]])),
                update='append',
                win='SRGAN SSIM',
                name='SRGAN SSIM'
            )

            train_vis.line(
                X=[NowStep],
                Y=[AllValPSNR],
                opts=dict(title='SRGAN PSNR',
                          linecolor=np.array([[255, 0, 0]])),
                update='append',
                win='SRGAN PSNR',
                name='SRGAN PSNR'
            )

            train_vis.line(
                X=np.array([[NowStep, NowStep]]),
                Y=np.array([[GLoss.item(), ValiLoss.item()]]),
                opts=dict(title='SRGAN GLoss',
                          linecolor=np.array([[255, 0, 0],
                                              [0, 0, 0]]),
                          markers=False,
                          legend=['GLoss', 'ValiLoss']),
                update='append',
                win='SRGAN GLoss'
            )
            train_vis.line(
                X=[NowStep],
                Y=[DLoss.item()],
                opts=dict(title='SRGAN DLoss',
                          linecolor=np.array([[255, 0, 0]])),
                update='append',
                win='SRGAN DLoss',
                name='SRGAN DLoss'
            )

    if epoch % 5 == 4:
        save_image(FakeImg.to(CPUDevice).data[:36], './Images/EpochSave/Epoch%d_Fake.jpg' % (epoch + 1), nrow=6)
        save_image(RealImg.to(CPUDevice).data[:36], './Images/EpochSave/Epoch%d_Real.jpg' % (epoch + 1), nrow=6)
        torch.save(obj=GModel.state_dict(), f='./Models/SRGAN.pth')

TestFake = GModel(TestX.to(GPUDevice)).to(CPUDevice)
SRGANImageQualityInfo = np.zeros((TestImageN, 2))
for ii in range(TestImageN):
    Img1 = np.transpose(TestY.data[ii].numpy(), (1, 2, 0))
    Img2 = np.transpose(TestX.data[ii].numpy(), (1, 2, 0))
    Img3 = np.transpose(TestFake.data[ii].numpy(), (1, 2, 0))
    cv2.imwrite('./Images/Real/Image%d.jpg' % (ii + 1), Img1 * 255)
    cv2.imwrite('./Images/Input/Image%d.jpg' % (ii + 1), Img2 * 255)
    cv2.imwrite('./Images/Fake/Image%d.jpg' % (ii + 1), Img3 * 255)
    FRSSIM = ssim(Img3, Img1, channel_axis=-1, data_range=1)
    FRPSNR = psnr(Img3, Img1, data_range=1)
    SRGANImageQualityInfo[ii, :] = [FRSSIM, FRPSNR]
np.save('SRGANImageQualityInfo', SRGANImageQualityInfo)
torch.save(obj=GModel.state_dict(), f='./Models/SRGAN.pth')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

imgpos = np.random.choice(100, 3)
infomat = np.load('SRGANImageQualityInfo.npy')
plt.figure()
for ii in range(3):
    img1 = cv2.imread('./Images/Real/Image%d.jpg' % (imgpos[ii] + 1), 1)
    img2 = cv2.imread('./Images/Input/Image%d.jpg' % (imgpos[ii] + 1), 1)
    img3 = cv2.imread('./Images/Fake/Image%d.jpg' % (imgpos[ii] + 1), 1)
    img2_us = cv2.resize(img2, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
    IRSSIM = ssim(img2_us, img1, data_range=255, channel_axis=-1)
    IRPSNR = psnr(img2_us, img1, data_range=255)

    plt.subplot(3, 3, ii + 1)
    plt.imshow(img1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Origin Image')

    plt.subplot(3, 3, ii + 4)
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])
    plt.title('DownSample Image,SSIM=%.3f,PSNR=%.1fdB' % (IRSSIM, IRPSNR))

    plt.subplot(3, 3, ii + 7)
    plt.imshow(img3)
    plt.xticks([])
    plt.yticks([])
    valssim = infomat[imgpos[ii], 0]
    valpsnr = infomat[imgpos[ii], 1]
    plt.title('SR Image,SSIM=%.3f,PSNR=%.1fdB' % (valssim, valpsnr))
plt.show()
