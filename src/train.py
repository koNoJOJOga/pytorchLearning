from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torchvision
from torch.utils.data import DataLoader
import torch
from model import *
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, 
                                        transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, 
                                        transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
ww = Ww()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(ww.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i+1))

    # 训练步骤开始
    ww.train()   # 有dropout、BN等层时需要加上
    for data in train_dataloader:
        imgs, targets = data
        outputs = ww(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 测试步骤开始
    ww.eval()   # 有dropout、BN等层时需要加上
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():   # 排除梯度的影响
        for data in test_dataloader:
            imgs, targets = data
            outputs = ww(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy +accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的acc：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(ww, "./model/ww_{}.pth".format(i))
    torch.save(ww.state_dict(), "./model/ww_{}.pth".format(i))
    print("模型已保存")

writer.close()


