# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.5.2
# ---

# # VGG
# 计算机视觉是一直深度学习的主战场，从这里我们将接触到近几年非常流行的卷积网络结构，网络结构由浅变深，参数越来越多，网络有着更多的跨层链接，首先我们先介绍一个数据集 cifar10，我们将以此数据集为例介绍各种卷积网络的结构。
#
# ## CIFAR 10
# cifar 10 这个数据集一共有 50000 张训练集，10000 张测试集，两个数据集里面的图片都是 png 彩色图片，图片大小是 32 x 32 x 3，一共是 10 分类问题，分别为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。这个数据集是对网络性能测试一个非常重要的指标，可以说如果一个网络在这个数据集上超过另外一个网络，那么这个网络性能上一定要比另外一个网络好，目前这个数据集最好的结果是 95% 左右的测试集准确率。
#
# ![](https://ws1.sinaimg.cn/large/006tNc79ly1fmpjxxq7wcj30db0ae7ag.jpg)
#
# 你能用肉眼对这些图片进行分类吗？
#
# cifar 10 已经被 pytorch 内置了，使用非常方便，只需要调用 `torchvision.datasets.CIFAR10` 就可以了

# ## VGGNet
# vggNet 是第一个真正意义上的深层网络结构，其是 ImageNet2014年的冠军，得益于 python 的函数和循环，我们能够非常方便地构建重复结构的深层网络。
#
# vgg 的网络结构非常简单，就是不断地堆叠卷积层和池化层，下面是一个简单的图示
#
# ![](https://ws4.sinaimg.cn/large/006tNc79ly1fmpk5smtidj307n0dx3yv.jpg)
#
# vgg 几乎全部使用 3 x 3 的卷积核以及 2 x 2 的池化层，使用小的卷积核进行多层的堆叠和一个大的卷积核的感受野是相同的，同时小的卷积核还能减少参数，同时可以有更深的结构。
#
# vgg 的一个关键就是使用很多层 3 x 3 的卷积然后再使用一个最大池化层，这个模块被使用了很多次，下面我们照着这个结构来写一写

# + {"ExecuteTime": {"start_time": "2017-12-22T09:01:50.883050Z", "end_time": "2017-12-22T09:01:51.296457Z"}}
import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
# -

# 我们可以定义一个 vgg 的 block，传入三个参数，第一个是模型层数，第二个是输入的通道数，第三个是输出的通道数，第一层卷积接受的输入通道就是图片输入的通道数，然后输出最后的输出通道数，后面的卷积接受的通道数就是最后的输出通道数

# + {"ExecuteTime": {"start_time": "2017-12-22T09:01:51.298777Z", "end_time": "2017-12-22T09:01:51.312500Z"}}
def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)] # 定义第一层
    
    for i in range(num_convs-1): # 定义后面的很多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
        
    net.append(nn.MaxPool2d(2, 2)) # 定义池化层
    return nn.Sequential(*net)
# -

# 我们可以将模型打印出来看看结构

# + {"ExecuteTime": {"start_time": "2017-12-22T08:20:40.808853Z", "end_time": "2017-12-22T08:20:40.819497Z"}}
block_demo = vgg_block(3, 64, 128)
print(block_demo)

# + {"ExecuteTime": {"start_time": "2017-12-22T07:52:02.381987Z", "end_time": "2017-12-22T07:52:04.632406Z"}}
# 首先定义输入为 (1, 64, 300, 300)
input_demo = Variable(torch.zeros(1, 64, 300, 300))
output_demo = block_demo(input_demo)
print(output_demo.shape)
# -

# 可以看到输出就变为了 (1, 128, 150, 150)，可以看到经过了这一个 vgg block，输入大小被减半，通道数变成了 128
#
# 下面我们定义一个函数对这个 vgg block 进行堆叠

# + {"ExecuteTime": {"start_time": "2017-12-22T09:01:54.489255Z", "end_time": "2017-12-22T09:01:54.497712Z"}}
def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)
# -

# 作为实例，我们定义一个稍微简单一点的 vgg 结构，其中有 8 个卷积层

# + {"ExecuteTime": {"start_time": "2017-12-22T09:01:55.041923Z", "end_time": "2017-12-22T09:01:55.149378Z"}}
vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
print(vgg_net)
# -

# 我们可以看到网络结构中有个 5 个 最大池化，说明图片的大小会减少 5 倍，我们可以验证一下，输入一张 256 x 256 的图片看看结果是什么

# + {"ExecuteTime": {"start_time": "2017-12-22T08:52:43.431478Z", "end_time": "2017-12-22T08:52:44.049650Z"}}
test_x = Variable(torch.zeros(1, 3, 256, 256))
test_y = vgg_net(test_x)
print(test_y.shape)
# -

# 可以看到图片减小了 $2^5$ 倍，最后再加上几层全连接，就能够得到我们想要的分类输出

# + {"ExecuteTime": {"start_time": "2017-12-22T09:01:57.306864Z", "end_time": "2017-12-22T09:01:57.323034Z"}}
class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
# -

# 然后我们可以训练我们的模型看看在 cifar10 上的效果

# + {"ExecuteTime": {"start_time": "2017-12-22T09:01:58.709531Z", "end_time": "2017-12-22T09:01:59.921373Z"}}
from utils import train

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1)) # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x
     
train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

net = vgg()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()

# + {"ExecuteTime": {"start_time": "2017-12-22T09:01:59.924086Z", "end_time": "2017-12-22T09:12:46.868967Z"}}
train(net, train_data, test_data, 20, optimizer, criterion)
# -

# 可以看到，跑完 20 次，vgg 能在 cifar 10 上取得 76% 左右的测试准确率
