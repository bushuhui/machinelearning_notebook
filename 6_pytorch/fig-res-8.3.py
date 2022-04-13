import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

#尝试用逻辑回归解决
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))

# [w,b]是模型的参数； 1e-1是学习速率
optimizer = torch.optim.SGD([w, b], 1e-1)
criterion = nn.BCEWithLogitsLoss()
def logistic_regression(x):
    return torch.mm(x, w) + b


for e in range(100):
    # 模型正向计算
    out = logistic_regression(Variable(x))
    # 计算误差
    loss = criterion(out, Variable(y))
    # 误差反传和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch:{}, loss:{}'.format(e+1, loss.item()))


def plot_decision_boundary(model, x, y):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1 
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max, h))    

    # Predict the function value for the whole grid .c_ 按行连接两个矩阵，左右相加。
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    for i in range(m):
        if y[i] == 0:
            plt.scatter(x[i, 0], x[i, 1], marker='8',c=0, s=40, cmap=plt.cm.Spectral)
        else:
            plt.scatter(x[i, 0], x[i, 1], marker='^',c=1, s=40)


def plot_logistic(x):
    x = Variable(torch.from_numpy(x).float())
    out = F.sigmoid(logistic_regression(x))
    out = (out > 0.5) * 1
    return out.data.numpy()

plot_decision_boundary(lambda x: plot_logistic(x), x.numpy(), y.numpy())
plt.title('逻辑回归')
plt.savefig('fig-res-8.3.pdf')
plt.show()