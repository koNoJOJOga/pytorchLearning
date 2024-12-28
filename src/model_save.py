import torchvision
import torch
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1，模型结构+模型参数
torch.save(vgg16, "./model/vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "./model/vgg16_method2.pth")

# 陷阱1
class wwwww(nn.Module):
    def __init__(self):
        super(wwwww, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

wei = wwwww()
torch.save(wei, "./model/wei_method1.pth")

