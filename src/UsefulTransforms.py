from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("images/test.jpg")
print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

# 如何使用生成文件？命令行：tensorboard --logdir=logs --port=6007

writer.add_image("ToTensor", img_tensor)
writer.close()