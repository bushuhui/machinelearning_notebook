import numpy as np

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt


"""
    Polynomial fitting by pytorch
"""

# define the real model's parameters
w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])

f_des = "y = %f + %f * x + %f * x^2 + %f * x^3" % (
    b_target[0],
    w_target[0], w_target[1], w_target[2])
print(f_des)

# draw the data
x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0]*x_sample + w_target[1]*x_sample**2 + w_target[2]*x_sample**3

plt.plot(x_sample, y_sample, label="Real")
plt.legend()
plt.show()


# construct variabels
x_train = np.stack([x_sample**i for i in range(1, 4)], axis=1)
x_train = torch.from_numpy(x_train).float()

y_train = torch.from_numpy(y_sample).float().unsqueeze(1)

# define model parameters
w = Variable(torch.randn(3, 1).float(), requires_grad=True)
b = Variable(torch.zeros(1).float(), requires_grad=True)

x_train = Variable(x_train)
y_train = Variable(y_train)


# define the model function & loss function
def polynomial(x):
    return torch.mm(x, w) + b

def get_loss(y_pred, y):
    return torch.mean((y_pred-y)**2)


# begin iterative optimization
eta = 0.001

for i in range(100):
    y_pred = polynomial(x_train)

    loss = get_loss(y_pred, y_train)
    loss.backward()

    w.data = w.data - eta*w.grad.data
    b.data = b.data - eta*b.grad.data

    w.grad.data.zero_()
    b.grad.data.zero_()

    if i % 10 == 0:
        print("epoch: %4d, loss: %f" % (i, loss.data[0]))

# draw the results
y_pred = polynomial(x_train)

plt.plot(x_train.data.numpy()[:, 0], y_sample, label="Real", color='b')
plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label="Fitting", color='r')
plt.legend()
plt.show()
