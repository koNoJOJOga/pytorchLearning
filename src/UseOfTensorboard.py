from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

writer.add_image()

# y = x^2
for i in range(100):
    writer.add_scalar(tag="y = x^2", scalar_value=i * i, global_step=i) # 标量的意思

# 如何使用生成文件？命令行：tensorboard --logdir=logs --port=6007

writer.close()