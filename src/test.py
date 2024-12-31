from PIL import Image
import torchvision
from model import *
import torch

image_path = "images/dog.png"
image = Image.open(image_path)

image = image.convert('RGB')    # 保留三颜色通道
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),    # 转换成32*32格式
    torchvision.transforms.ToTensor()           # 转换成tensor
])

image = transform(image)

model = torch.load
vgg16.load_state_dict(torch.load("./model/vgg16_method2.pth"))
model = torch.load("vgg16_method2.pth")