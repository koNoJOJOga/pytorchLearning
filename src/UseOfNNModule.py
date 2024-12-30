import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset/", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class ww(nn.Module):
    def __init__(self):
        super(ww, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

www = ww()

writer = SummaryWriter("dataloader/")
step = 0
for data in dataloader:
    imgs, target = data
    output = www(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1

# 启动tensorboard: tensorboard --logdir=dataloader
