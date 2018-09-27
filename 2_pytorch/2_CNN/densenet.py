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

# # DenseNet
# 因为 ResNet 提出了跨层链接的思想，这直接影响了随后出现的卷积网络架构，其中最有名的就是 cvpr 2017 的 best paper，DenseNet。
#
# DenseNet 和 ResNet 不同在于 ResNet 是跨层求和，而 DenseNet 是跨层将特征在通道维度进行拼接，下面可以看看他们两者的图示
#
# ![](https://ws4.sinaimg.cn/large/006tNc79ly1fmpvj5vkfhj30uw0anq73.jpg)
#
# ![](https://ws1.sinaimg.cn/large/006tNc79ly1fmpvj7fxd1j30vb0eyzqf.jpg)

# 第一张图是 ResNet，第二张图是 DenseNet，因为是在通道维度进行特征的拼接，所以底层的输出会保留进入所有后面的层，这能够更好的保证梯度的传播，同时能够使用低维的特征和高维的特征进行联合训练，能够得到更好的结果。

# DenseNet 主要由 dense block 构成，下面我们来实现一个 densen block

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:30.612922Z", "end_time": "2017-12-22T15:38:31.113030Z"}}
import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
# -

# 首先定义一个卷积块，这个卷积块的顺序是 bn -> relu -> conv

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.115369Z", "end_time": "2017-12-22T15:38:31.121249Z"}}
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer
# -

# dense block 将每次的卷积的输出称为 `growth_rate`，因为如果输入是 `in_channel`，有 n 层，那么输出就是 `in_channel + n * growh_rate`

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.123363Z", "end_time": "2017-12-22T15:38:31.145274Z"}}
class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
            
        self.net = nn.Sequential(*block)
        
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x
# -

# 我们验证一下输出的 channel 是否正确

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.147196Z", "end_time": "2017-12-22T15:38:31.213632Z"}}
test_net = dense_block(3, 12, 3)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input shape: {} x {} x {}'.format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))
# -

# 除了 dense block，DenseNet 中还有一个模块叫过渡层（transition block），因为 DenseNet 会不断地对维度进行拼接， 所以当层数很高的时候，输出的通道数就会越来越大，参数和计算量也会越来越大，为了避免这个问题，需要引入过渡层将输出通道降低下来，同时也将输入的长宽减半，这个过渡层可以使用 1 x 1 的卷积

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.215770Z", "end_time": "2017-12-22T15:38:31.222120Z"}}
def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer
# -

# 验证一下过渡层是否正确

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.224078Z", "end_time": "2017-12-22T15:38:31.234846Z"}}
test_net = transition(3, 12)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input shape: {} x {} x {}'.format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))
# -

# 最后我们定义 DenseNet

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.236857Z", "end_time": "2017-12-22T15:38:31.318822Z"}}
class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(densenet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )
        
        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition(channels, channels // 2)) # 通过 transition 层将大小减半，通道数减半
                channels = channels // 2
        
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        
        self.classifier = nn.Linear(channels, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.320788Z", "end_time": "2017-12-22T15:38:31.654182Z"}}
test_net = densenet(3, 10)
test_x = Variable(torch.zeros(1, 3, 96, 96))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:31.656356Z", "end_time": "2017-12-22T15:38:32.894729Z"}}
from utils import train

def data_tf(x):
    x = x.resize((96, 96), 2) # 将图片放大到 96 x 96
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1)) # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x
     
train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

net = densenet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# + {"ExecuteTime": {"start_time": "2017-12-22T15:38:32.896735Z", "end_time": "2017-12-22T16:15:38.168095Z"}}
train(net, train_data, test_data, 20, optimizer, criterion)
# -

# DenseNet 将残差连接改为了特征拼接，使得网络有了更稠密的连接
