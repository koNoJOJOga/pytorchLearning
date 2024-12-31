from PIL import Image
import torchvision
from model import *
import torch

image_path = "images/airplane.png"
image = Image.open(image_path)

image = image.convert('RGB')    # 保留三颜色通道
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),    # 转换成32*32格式
    torchvision.transforms.ToTensor()           # 转换成tensor
])

image = transform(image)

# 创建模型实例
model = Ww()

# 加载保存的状态字典
model.load_state_dict(torch.load("./model/best_model.pth"))

# 将模型转到需要的设备
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model.to(device)

image = torch.reshape(image, (1, 3, 32, 32))

# 设置为评估模式
model.eval()

with torch.no_grad():
    output = model(image)
print(output)


print(output.argmax(1))