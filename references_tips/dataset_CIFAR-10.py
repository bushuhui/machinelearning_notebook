
import os

import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化

# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约100M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])

# set data storage dir & check whether do download
data_path = "../data/"
p = os.path.join(data_path, "cifar-10-batches-py")
do_download = True
if os.path.isdir(p):
    do_download = False

# 训练集
trainset = tv.datasets.CIFAR10(
                    root=data_path,
                    train=True,
                    download=do_download,
                    transform=transform)

trainloader = t.utils.data.DataLoader(
                    trainset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
                    root=data_path,
                    train=False,
                    download=do_download,
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
