
import numpy as np

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

"""
Using pytorch to do linear regression
"""

torch.manual_seed(2018)

# generate data
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# draw the data
plt.plot(x_train, y_train, 'bo')
plt.show()


# convert to tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# define model parameters
w = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

# construct the linear model
x_train = Variable(x_train)
y_train = Variable(y_train)

def linear_model(x):
    return x*w + b

# first predictive
y_pred = linear_model(x_train)

# draw the real & predictived data
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label="Real")
plt.plot(x_train.data.numpy(), y_pred.data.numpy(), 'ro', label="Estimated")
plt.legend()
plt.show()

# define the loss function
def get_loss(y_pred, y):
    return torch.mean((y_pred - y)**2)

loss = get_loss(y_pred, y_train)
print("loss = %f" % float(loss))


# auto-grad
loss.backward()
print("w.grad = %f" % float(w.grad))
print("b.grad = %f" % float(b.grad))

# upgrade parameters
eta = 1e-2

w.data = w.data - eta*w.grad.data
b.data = b.data - eta*w.grad.data

y_pred = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label="Real")
plt.plot(x_train.data.numpy(), y_pred.data.numpy(), 'ro', label="Estimated")
plt.legend()
plt.show()


for i in range(10):
    y_pred = linear_model(x_train)
    loss = get_loss(y_pred, y_train)

    w.grad.zero_()
    b.grad.zero_()
    loss.backward()

    w.data = w.data - eta*w.grad.data
    b.data = b.data - eta*b.grad.data

    print("epoch: %3d, loss: %f" % (i, loss.data[0]))

