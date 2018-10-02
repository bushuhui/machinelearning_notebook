import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F


# generate sample data
centers = [(0, 0), (5, 5)]
n_samples = 200

x_train, y_train = datasets.make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
y_label = y_train

# plot data
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_label, label="Real", cmap=plt.cm.Spectral)
plt.show()

# convert to tensor
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
y_train.unsqueeze_(1)

# define model parameters
w = Variable(torch.randn(2, 1).float(), requires_grad=True)
b = Variable(torch.zeros(1).float(), requires_grad=True)

# construct the linear model
x_train = Variable(x_train)
y_train = Variable(y_train)

# define logistic regression function
def logistic_regression(x):
    return torch.sigmoid(torch.mm(x, w) + b)

# define loss function
def binary_loss(y_pred, y):
    logits = (y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return -logits

# upgrade parameters
eta = 1e-2
n_epoch = 1000

for i in range(n_epoch):
    y_pred = logistic_regression(x_train)

    loss = binary_loss(y_pred, y_train)
    loss.backward()

    w.data = w.data - eta*w.grad.data
    b.data = b.data - eta*b.grad.data

    w.grad.zero_()
    b.grad.zero_()

    y_est = y_pred.ge(0.5).float()
    acc = float((y_est == y_train).sum().data[0]) / y_train.shape[0]
    if i % 10 == 0:
        print("epoch: %3d, loss: %f, acc: %f" % (i, loss.data[0], acc))


# plot decision boundary
w0 = float(w[0].data[0])
w1 = float(w[1].data[0])
b0 = float(b.data[0])
print("w: %f %f, b = %f" % (w0, w1, b0))

x_min = float(x_train[:, 0].min())
x_max = float(x_train[:, 0].max())
plot_x = np.arange(x_min, x_max, 0.1)
plot_y = (-w0*plot_x - b0)/w1

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_label, label="Real", cmap=plt.cm.Spectral)
plt.plot(plot_x, plot_y, 'g-', label="Decision boundary")
plt.legend()
plt.show()

y_pred = logistic_regression(x_train)
y_est = torch.Tensor(y_pred.size())
y_est[y_pred > 0.5] = 1
y_est[y_pred < 0.5] = 0

y_est = y_est.numpy().flatten()
err = np.sum((y_est - y_label)**2)
print("err = %f" % err)

