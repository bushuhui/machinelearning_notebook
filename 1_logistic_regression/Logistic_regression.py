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

# # Logistic Regression
#
# 逻辑回归(Logistic Regression, LR)模型其实仅在线性回归的基础上，套用了一个逻辑函数，但也就由于这个逻辑函数，使得逻辑回归模型成为了机器学习领域一颗耀眼的明星，更是计算广告学的核心。本节主要详述逻辑回归模型的基础。
#
#
# ## 1 逻辑回归模型
# 回归是一种比较容易理解的模型，就相当于$y=f(x)$，表明自变量$x$与因变量$y$的关系。最常见问题有如医生治病时的望、闻、问、切，之后判定病人是否生病或生了什么病，其中的望闻问切就是获取自变量$x$，即特征数据，判断是否生病就相当于获取因变量$y$，即预测分类。
#
# 最简单的回归是线性回归，在此借用Andrew NG的讲义，有如图所示，$X$为数据点——肿瘤的大小，$Y$为观测值——是否是恶性肿瘤。通过构建线性回归模型，如$h_\theta(x)$所示，构建线性回归模型后，即可以根据肿瘤大小，预测是否为恶性肿瘤$h_\theta(x)) \ge 0.5$为恶性，$h_\theta(x) \lt 0.5$为良性。
#
# ![LinearRegression](images/fig1.gif)
#
# 然而线性回归的鲁棒性很差，例如在上图的数据集上建立回归，因最右边噪点的存在，使回归模型在训练集上表现都很差。这主要是由于线性回归在整个实数域内敏感度一致，而分类范围，需要在$[0,1]$。
#
# 逻辑回归就是一种减小预测范围，将预测值限定为$[0,1]$间的一种回归模型，其回归方程与回归曲线如图2所示。逻辑曲线在$z=0$时，十分敏感，在$z>>0$或$z<<0$处，都不敏感，将预测值限定为$(0,1)$。
#
# ![LogisticFunction](images/fig2.gif)
#
#

# +
# %matplotlib inline

from __future__ import division
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

np.random.seed(0)


# +
# load sample data
data, label = sklearn.datasets.make_moons(200, noise=0.30)

print("data  = ", data[:10, :])
print("label = ", label[:10])

plt.scatter(data[:,0], data[:,1], c=label)
plt.title("Original Data")

# +
def plot_decision_boundary(predict_func, data, label):
    """画出结果图
    Args:
        pred_func (callable): 预测函数
        data (numpy.ndarray): 训练数据集合
        label (numpy.ndarray): 训练数据标签
    """
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Spectral)
    plt.show()



# +
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class Logistic(object):
    """logistic回归模型"""
    def __init__(self, data, label):
        self.data = data
        self.label = label

        self.data_num, n = np.shape(data)
        self.weights = np.ones(n)
        self.b = 1

    def train(self, num_iteration=150):
        """随机梯度上升算法
        Args:
            data (numpy.ndarray): 训练数据集
            labels (numpy.ndarray): 训练标签
            num_iteration (int): 迭代次数
        """
        for j in range(num_iteration):
            data_index = list(range(self.data_num))
            for i in range(self.data_num):
                # 学习速率
                alpha = 0.01
                rand_index = int(np.random.uniform(0, len(data_index)))
                error = self.label[rand_index] - sigmoid(sum(self.data[rand_index] * self.weights + self.b))
                self.weights += alpha * error * self.data[rand_index]
                self.b += alpha * error
                del(data_index[rand_index])

    def predict(self, predict_data):
        """预测函数"""
        result = list(map(lambda x: 1 if sum(self.weights * x + self.b) > 0 else 0,
                     predict_data))
        return np.array(result)

# -

logistic = Logistic(data, label)
logistic.train(200)
plot_decision_boundary(lambda x: logistic.predict(x), data, label)

# ## References
#
# * [逻辑回归模型(Logistic Regression, LR)基础](https://www.cnblogs.com/sparkwen/p/3441197.html)
# * [逻辑回归（Logistic Regression）](http://www.cnblogs.com/BYRans/p/4713624.html)
