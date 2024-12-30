from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torchvision
from torch.utils.data import DataLoader

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
for data in dataloader:
    imgs, targets = data
    output = www(imgs)
    result_loss = loss(output, targets) # 使用损失函数计算出误差
    result_loss.backward()  # 对梯度进行反向传播得到梯度
