import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%matplotlib inline
np.random.seed(1)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量， 0 表示红色， 1 表示蓝色
a = 4

# 生成两类数据
for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.ylabel("x2")
plt.xlabel("x1")
# 绘出生成的数据
for i in range(m):
    if y[i] == 0:
        plt.scatter(x[i, 0], x[i, 1], marker='8',c=0, s=40, cmap=plt.cm.Spectral)
    else:
        plt.scatter(x[i, 0], x[i, 1], marker='^',c=1, s=40)
plt.savefig('fig-res-8.2.pdf')
plt.show()
# #尝试用逻辑回归解决
# x = torch.from_numpy(x).float()
# y = torch.from_numpy(y).float()

# w = nn.Parameter(torch.randn(2, 1))
# b = nn.Parameter(torch.zeros(1))

# # [w,b]是模型的参数； 1e-1是学习速率
# optimizer = torch.optim.SGD([w, b], 1e-1)
# criterion = nn.BCEWithLogitsLoss()
# def logistic_regression(x):
#     return torch.mm(x, w) + b


# for e in range(100):
#     # 模型正向计算
#     out = logistic_regression(Variable(x))
#     # 计算误差
#     loss = criterion(out, Variable(y))
#     # 误差反传和参数更新
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (e + 1) % 20 == 0:
#         print('epoch:{}, loss:{}'.format(e+1, loss.item()))


# def plot_decision_boundary(model, x, y):
#     # Set min and max values and give it some padding
#     x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
#     y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1 
#     h = 0.01
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max, h))    

#     # Predict the function value for the whole grid .c_ 按行连接两个矩阵，左右相加。
#     Z = model(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.ylabel("x2")
#     plt.xlabel("x1")
#     plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)

# def plot_logistic(x):
#     x = Variable(torch.from_numpy(x).float())
#     out = F.sigmoid(logistic_regression(x))
#     out = (out > 0.5) * 1
#     return out.data.numpy()

# plot_decision_boundary(lambda x: plot_logistic(x), x.numpy(), y.numpy())
# plt.title('logistic regression')
# plt.savefig('fig-res-8.3.pdf')


# # 定义两层神经网络的参数
# w1 = nn.Parameter(torch.randn(2, 4) * 0.01) # 输入维度为2， 隐藏层神经元个数4
# b1 = nn.Parameter(torch.zeros(4))
# w2 = nn.Parameter(torch.randn(4, 1) * 0.01) # 隐层神经元为4， 输出单元为1
# b2 = nn.Parameter(torch.zeros(1))

# def mlp_network(x):
#     x1 = torch.mm(x, w1) + b1
#     x1 = F.tanh(x1) # 使用 PyTorch 自带的 tanh 激活函数
#     x2 = torch.mm(x1, w2) + b2
#     return x2

# # 定义优化器和损失函数
# optimizer = torch.optim.SGD([w1, w2, b1, b2], 1.)
# criterion = nn.BCEWithLogitsLoss()

# for e in range(10000):
#     # 正向计算
#     out = mlp_network(Variable(x))
#     # 计算误差
#     loss = criterion(out, Variable(y))
#     # 计算梯度并更新权重
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (e + 1) % 1000 == 0:
#         print('epoch: {}, loss: {}'.format(e+1, loss.item()))

# def plot_network(x):
#     x = Variable(torch.from_numpy(x).float())
#     x1 = torch.mm(x, w1) + b1
#     x1 = F.tanh(x1)
#     x2 = torch.mm(x1, w2) + b2
#     out = F.sigmoid(x2)
#     out = (out > 0.5) * 1
#     return out.data.numpy()

# plot_decision_boundary(lambda x: plot_network(x), x.numpy(), y.numpy())
# plt.title('2 layer network')
# plt.savefig('fig-res-8.4.pdf')

# # Sequential
# seq_net = nn.Sequential(
#     nn.Linear(2, 4), # PyTorch 中的线性层， wx + b
#     nn.Tanh(),
#     nn.Linear(4, 1)
# )

# # 序列模块可以通过索引访问每一层
# seq_net[0] # 第一层

# # 打印出第一层的权重
# w0 = seq_net[0].weight
# print(w0)


# # 通过 parameters 可以取得模型的参数
# param = seq_net.parameters()
# # 定义优化器
# optim = torch.optim.SGD(param, 1.)

# # 训练 10000 次
# for e in range(10000):
#     # 网络正向计算
#     out = seq_net(Variable(x))
#     # 计算误差
#     loss = criterion(out, Variable(y))
#     # 反向传播、 更新权重
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     # 打印损失
#     if (e + 1) % 1000 == 0:
#         print('epoch: {}, loss: {}'.format(e+1, loss.item()))


# def plot_seq(x):
#     out = F.sigmoid(seq_net(Variable(torch.from_numpy(x).float()))).data.numpy()
#     out = (out > 0.5) * 1
#     return out

# plot_decision_boundary(lambda x: plot_seq(x), x.numpy(), y.numpy())
# plt.title('sequential')  
# plt.savefig('fig-res-8.5.pdf')

# torch.save(seq_net, 'save_seq_net.pth')      

# # 读取保存的模型
# seq_net1 = torch.load('save_seq_net.pth')
# # 打印加载的模型
# seq_net1

# print(seq_net1[0].weight)

# # 保存模型参数
# torch.save(seq_net.state_dict(), 'save_seq_net_params.pth')    

# # 定义网络架构
# seq_net2 = nn.Sequential(
#     nn.Linear(2, 4),
#     nn.Tanh(),
#     nn.Linear(4, 1)
# )
# # 加载网络参数
# seq_net2.load_state_dict(torch.load('save_seq_net_params.pth'))

# # 打印网络结构
# seq_net2
# print(seq_net2[0].weight)

# class Module_Net(nn.Module):
#     def __init__(self, num_input, num_hidden, num_output):
#         super(Module_Net, self).__init__()
#         self.layer1 = nn.Linear(num_input, num_hidden)
#         self.layer2 = nn.Tanh()
#         self.layer3 = nn.Linear(num_hidden, num_output)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x

# mo_net = Module_Net(2, 4, 1)
# # 访问模型中的某层可以直接通过名字， 网络第一层
# l1 = mo_net.layer1
# print(l1)


# optim = torch.optim.SGD(mo_net.parameters(), 1.)
#     # 训练 10000 次
# for e in range(10000):
#     # 网络正向计算
#     out = mo_net(Variable(x))
#     # 计算误差
#     loss = criterion(out, Variable(y))
#     # 误差反传、 更新参数
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     # 打印损失
#     if (e + 1) % 1000 == 0:
#         print('epoch: {}, loss: {}'.format(e+1, loss.item()))

# torch.save(mo_net.state_dict(), 'module_net.pth')




