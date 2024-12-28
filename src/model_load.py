import torch
import torchvision
from model_save import *

# 方式2-->保存方式1，加载模型
model = torch.load("./model/vgg16_method1.pth", weights_only=False)
# print(model)

# 方式2-->保存方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./model/vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1
model = torch.load("./model/wei_method1.pth", weights_only=False)
print(model)

