import torch
import torch.utils.data as Data
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
import sys
import time
from time import strftime, gmtime
import visdom

sys.path.append('../ESRGAN/')
from LoadDataSet import LoadHRLRImage
from CreateModel import SRGANGenerator, SRGANDiscriminator
from skimage.metrics import peak_signal_noise_ratio as psnr
from Loss import AllLoss

TrainVis = visdom.Visdom(env='SRGAN')
GPUDevice = torch.device("cuda")
CPUDevice = torch.device("cpu")

HRData, LRData = LoadHRLRImage(DataType='Train')
TrainData = Data.TensorDataset(HRData, LRData)

ValidHRImg, ValidLRImg = LoadHRLRImage(DataType='Valid')
AllValidN = ValidHRImg.shape[0]

ValidN = 25
TestN = AllValidN - ValidN

ValidHRData = ValidHRImg[:ValidN]
ValidLRData = ValidLRImg[:ValidN]
TestHRData = ValidHRImg[ValidN:AllValidN]
TestLRData = ValidLRImg[ValidN:AllValidN]

BatchSize = 16
TrainLoader = Data.DataLoader(
    dataset=TrainData,
    shuffle=True,
    drop_last=True,
    batch_size=BatchSize
)
# ----------------------------------PSNR训练----------------------------
GModel = SRGANGenerator().to(GPUDevice)
MSELoss = torch.nn.MSELoss(reduction='mean')
GOptim = optim.Adam(GModel.parameters(), lr=1e-4, betas=(0.9, 0.999))
MaxEpochPSNR = 2000
SingleEpochIterNPSNR = int(HRData.shape[0] / BatchSize)
MaxIterPSNR = SingleEpochIterNPSNR * MaxEpochPSNR
NowStepPSNR = 0

TrainVis.close(win='SRGAN GLoss')
TrainVis.close(win='SRGAN PSNR')
StartTime = time.time()
for epoch in range(MaxEpochPSNR):
    for step, (RealImg, InputImg) in enumerate(TrainLoader):
        NowStepPSNR = NowStepPSNR + 1
        RealImg = RealImg.to(GPUDevice)
        InputImg = InputImg.to(GPUDevice)
        FakeImg = GModel(InputImg)
        Loss = MSELoss(FakeImg, RealImg)
        GOptim.zero_grad()
        Loss.backward()
        GOptim.step()
        if step % 20 == 0:
            GModel.eval()
            val_fake_img = GModel(ValidLRData.to(GPUDevice))
            GModel.train()
            ValidLoss = MSELoss(val_fake_img, ValidHRData.to(GPUDevice))
            val_fake_img = val_fake_img.to(CPUDevice)
            torch.cuda.empty_cache()
            all_psnr = 0
            for val_id in range(ValidN):
                real_img = np.transpose(ValidHRData.data[val_id].numpy(), (1, 2, 0))
                fake_img = np.transpose(val_fake_img.data[val_id].numpy(), (1, 2, 0))
                val_psnr = psnr(fake_img, real_img, data_range=1)
                all_psnr = all_psnr + val_psnr
            all_psnr = all_psnr / ValidN

            TrainVis.line(
                X=np.array([[NowStepPSNR, NowStepPSNR]]),
                Y=np.array([[Loss.item() / BatchSize, ValidLoss.item() / ValidN]]),
                opts=dict(
                    title='GLoss',
                    linecolor=np.array([[255, 0, 0],
                                        [0, 0, 0]]),
                    legend=['Train', 'Valid']
                ),
                win='SRGAN GLoss',
                update='append'
            )

            TrainVis.line(
                X=[NowStepPSNR],
                Y=[all_psnr],
                opts=dict(
                    title='PSNR',
                    linecolor=np.array([[255, 0, 0]])
                ),
                win='SRGAN PSNR',
                name='SRGAN PSNR',
                update='append'
            )
            Duration = time.time() - StartTime
            IterSpeed = NowStepPSNR / Duration
            AllTakeTime = Duration + (MaxIterPSNR - NowStepPSNR) / IterSpeed
            NowPercentage = 100 * NowStepPSNR / MaxIterPSNR
            print('PSNR %.2f%%:Iter=%d/%d,Epoch=%d/%d,' % (NowPercentage, NowStepPSNR, MaxIterPSNR, epoch + 1, MaxEpochPSNR),
                  'GLoss=%.4f,' % (Loss.item() / BatchSize),
                  strftime("%H:%M:%S", gmtime(Duration)),
                  '/',
                  strftime("%H:%M:%S", gmtime(AllTakeTime)),
                  ',%.1fiter/s,' % IterSpeed,
                  'PSNR=%.2fdB' % all_psnr)
    if epoch % 5 == 4:
        torch.save(obj=GModel.state_dict(), f='./Models/ModelPSNR.pth')
        save_image(FakeImg.to(CPUDevice).data, './Images/EpochSavePSNR/Epoch%d_Fake.jpg' % (epoch + 1), nrow=4)
        save_image(RealImg.to(CPUDevice).data, './Images/EpochSavePSNR/Epoch%d_Real.jpg' % (epoch + 1), nrow=4)

torch.save(obj=GModel.state_dict(), f='./Models/ModelPSNR.pth')

# --------------------------------对抗训练-----------------------------------
# GModel = SRGANGenerator().to(GPUDevice)
# GState = torch.load('./Models/ModelPSNR.pth')
# GModel.load_state_dict(GState)
# DModel = SRGANDiscriminator().to(GPUDevice)
# GOptim = optim.Adam(GModel.parameters(), lr=1e-4, betas=(0.9, 0.999))
# DOptim = optim.Adam(DModel.parameters(), lr=1e-4, betas=(0.9, 0.999))
# GLROpt = optim.lr_scheduler.StepLR(GOptim, step_size=100, gamma=0.1)
# DLROpt = optim.lr_scheduler.StepLR(DOptim, step_size=100, gamma=0.1)
# GLossFunc = AllLoss(GPUDevice)
# DLossFunc = torch.nn.BCELoss().to(GPUDevice)
# MaxEpochGAN = 200
# MaxIterGAN = MaxEpochGAN*int(HRData.shape[0]/BatchSize)
# NowStepGAN = 0
# StartTime = time.time()
# TrainVis.close(win='SRGAN GLoss')
# TrainVis.close(win='SRGAN DLoss')
# TrainVis.close(win='SRGAN PSNR')
# for epoch in range(MaxEpochGAN):
#     for step, (RealImg, InputImg) in enumerate(TrainLoader):
#         NowStepGAN = NowStepGAN + 1
#         RealImg = RealImg.to(GPUDevice)
#         InputImg = InputImg.to(GPUDevice)
#         FakeImg = GModel(InputImg)
#         FakeLabel = DModel(FakeImg)
#         RealLabel = DModel(RealImg)
#         TrueLabel = torch.tensor(np.ones((BatchSize, 1)), dtype=torch.float32).to(GPUDevice)
#         FalseLabel = torch.tensor(np.zeros((BatchSize, 1)), dtype=torch.float32).to(GPUDevice)
#         DLoss = (DLossFunc(RealLabel, TrueLabel) + DLossFunc(FakeLabel, FalseLabel)) / 2
#         if NowStepGAN % 3 == 1:
#             DOptim.zero_grad()
#             DLoss.backward(retain_graph=True)
#             DOptim.step()
#
#         FakeLabel = DModel(FakeImg)
#         GLoss = GLossFunc(FakeImg, RealImg, FakeLabel)
#         GOptim.zero_grad()
#         GLoss.backward()
#         GOptim.step()
#         if step % 20 == 0:
#             GModel.eval()
#             val_fake_img = GModel(ValidLRData.to(GPUDevice))
#             val_fake_label = DModel(val_fake_img)
#             val_fake_img = val_fake_img.to(CPUDevice)
#             all_psnr = 0
#             for val_id in range(ValidN):
#                 real_img = np.transpose(ValidHRData.data[val_id].numpy(), (1, 2, 0))
#                 fake_img = np.transpose(val_fake_img.data[val_id].numpy(), (1, 2, 0))
#                 val_psnr = psnr(fake_img, real_img, data_range=1)
#                 all_psnr = all_psnr + val_psnr
#             all_psnr = all_psnr / ValidN
#
#             val_fake_img = val_fake_img.to(GPUDevice)
#             val_real_img = ValidHRData.to(GPUDevice)
#             val_real_label = DModel(val_real_img)
#             ValiGLoss = GLossFunc(val_fake_img, val_real_img, val_fake_label)
#             GModel.train()
#
#             TrainVis.line(
#                 X=np.array([[NowStepGAN, NowStepGAN]]),
#                 Y=np.array([[GLoss.item(), ValiGLoss.item()]]),
#                 opts=dict(
#                     title='GLoss',
#                     linecolor=np.array([[255, 0, 0],
#                                         [0, 0, 0]]),
#                     legend=['Train', 'Valid']
#                 ),
#                 win='SRGAN GLoss',
#                 update='append'
#             )
#
#             TrainVis.line(
#                 X=[NowStepGAN],
#                 Y=[DLoss.item() / BatchSize],
#                 opts=dict(
#                     title='DLoss',
#                     linecolor=np.array([[255, 0, 0]])
#                 ),
#                 win='SRGAN DLoss',
#                 name='SRGAN DLoss',
#                 update='append'
#             )
#
#             TrainVis.line(
#                 X=[NowStepGAN],
#                 Y=[all_psnr],
#                 opts=dict(
#                     title='PSNR',
#                     linecolor=np.array([[255, 0, 0]])
#                 ),
#                 win='SRGAN PSNR',
#                 name='SRGAN PSNR',
#                 update='append'
#             )
#
#             Duration = time.time() - StartTime
#             IterSpeed = NowStepGAN / Duration
#             AllTakeTime = Duration + (MaxIterGAN - NowStepGAN) / IterSpeed
#             NowPercentage = 100 * NowStepGAN / MaxIterGAN
#             print('GAN %.2f%%:Iter=%d/%d,Epoch=%d/%d,' % (NowPercentage, NowStepGAN, MaxIterGAN, epoch + 1, MaxEpochGAN),
#                   'GLoss=%.4f,' % GLoss.item(),
#                   'DLoss=%.4f,' % DLoss.item(),
#                   strftime("%H:%M:%S", gmtime(Duration)),
#                   '/',
#                   strftime("%H:%M:%S", gmtime(AllTakeTime)),
#                   ',%.1fiter/s' % IterSpeed,
#                   ',PSNR=%.2fdB' % all_psnr)
#     GLROpt.step()
#     DLROpt.step()
#     if epoch % 5 == 4:
#         torch.save(obj=GModel.state_dict(), f='./Models/ModelGAN.pth')
#         save_image(FakeImg.to(CPUDevice).data, './Images/EpochSaveGAN/Epoch%d_Fake.jpg' % (epoch+1), nrow=4)
#         save_image(RealImg.to(CPUDevice).data, './Images/EpochSaveGAN/Epoch%d_Real.jpg' % (epoch + 1), nrow=4)
#
# torch.save(obj=GModel.state_dict(), f='./Models/ModelGAN.pth')