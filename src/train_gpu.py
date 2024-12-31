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

# 想要使用gpu，需要将模型和输入数据（如 imgs 和 targets）移动到 mps。
# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 创建网络模型并移动到设备
ww = Ww().to(device)

# 损失函数移动到设备
loss_fn = nn.CrossEntropyLoss().to(device)

# 定义优化器为 AdamW
learning_rate = 0.01
optimizer = torch.optim.AdamW(ww.parameters(), lr=learning_rate, weight_decay=0.01)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10

# 添加 TensorBoard
writer = SummaryWriter("logs_train")

# Early Stopping 参数
early_stop_patience = 3  # 验证集性能连续多少轮没有提升时停止
no_improvement_count = 0  # 用于计数验证集性能没有提升的轮次

best_accuracy = 0.0
best_epoch = 0

for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i+1))

    # 训练步骤开始
    ww.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = ww(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 每个 epoch 结束后进行验证
    ww.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = ww(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    avg_accuracy = total_accuracy / test_data_size
    print("当前验证集上的acc：{}".format(avg_accuracy))
    writer.add_scalar("test_accuracy", avg_accuracy, total_test_step)
    total_test_step += 1

    # 判断是否更新最佳模型
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_epoch = i
        torch.save(ww.state_dict(), "./model/best_model.pth")
        print(f"保存最好的模型，epoch {i+1}，acc {avg_accuracy:.4f}")
        no_improvement_count = 0  # 重置计数
    else:
        no_improvement_count += 1
        print(f"验证集性能未提升，连续未提升次数：{no_improvement_count}/{early_stop_patience}")

    # 判断是否触发 Early Stopping
    if no_improvement_count >= early_stop_patience:
        print(f"Early stopping at epoch {i+1}. 最佳模型在第 {best_epoch+1} 轮获得，准确率：{best_accuracy:.4f}")
        break


# 训练完成后，显示最佳模型信息
print(f"最佳模型是在第 {best_epoch+1} 轮获得的，准确率：{best_accuracy:.4f}")
writer.close()


