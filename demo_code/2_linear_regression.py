
import numpy as np

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

"""
Using pytorch to do linear regression
"""

torch.manual_seed(2018)

# model's real-parameters
w_target = 3
b_target = 10

# generate data
n_data = 100
x_train = np.random.rand(n_data, 1)*20 - 10
y_train = w_target*x_train + b_target + (np.random.randn(n_data, 1)*10-5.0)

# draw the data
plt.plot(x_train, y_train, 'bo')
plt.show()


# convert to tensor
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

# define model parameters
w = Variable(torch.randn(1).float(), requires_grad=True)
b = Variable(torch.zeros(1).float(), requires_grad=True)

# construct the linear model
x_train = Variable(x_train)
y_train = Variable(y_train)


# define model's function
def linear_model(x):
    return x*w + b

# define the loss function
def get_loss(y_pred, y):
    return torch.mean((y_pred - y)**2)

# upgrade parameters
eta = 1e-2

for i in range(100):
    y_pred = linear_model(x_train)

    loss = get_loss(y_pred, y_train)
    loss.backward()

    w.data = w.data - eta*w.grad.data
    b.data = b.data - eta*b.grad.data

    w.grad.zero_()
    b.grad.zero_()

    if i % 10 == 0:
        print("epoch: %3d, loss: %f" % (i, loss.data[0]))

# draw the results
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label="Real")
plt.plot(x_train.data.numpy(), y_pred.data.numpy(), 'ro', label="Estimated")
plt.legend()
plt.show()