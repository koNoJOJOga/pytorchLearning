from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torchvision
from torch.utils.data import DataLoader
import torch

dataset = torchvision.datasets.CIFAR10("dataset/", train=False, transform=torchvision.transforms.ToTensor(), download=True) # 下载数据集
dataloader = DataLoader(dataset, batch_size=64) # DataLoader进行加载

class ww(nn.Module):
    def __init__(self):
        super(ww, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
www = ww()
optim = torch.optim.SGD(www.parameters(), lr=0.01, )    # 开始时使用大学习速率，后期用相对小的
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = www(imgs)
        result_loss = loss(output, targets) # 使用损失函数计算出误差
        # 可以debug接下来的三行，找到对应模型、modules、model1、modules、weight、grad，观察梯度变化和参数变化
        optim.zero_grad()   # 要记得梯度归零，防止上一层梯度的影响
        result_loss.backward()  # 对梯度进行反向传播得到梯度
        optim.step()        # 优化器优化，对参数进行调整
        running_loss = running_loss + result_loss   # 整体误差的总和
    print(running_loss)
