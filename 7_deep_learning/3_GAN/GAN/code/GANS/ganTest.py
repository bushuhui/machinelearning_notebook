"""
function: GAN show demo
data :2021-08-26
author:zuo_kui_jun
"""
import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns  # Seaborn是比Matplotlib更高级的免费库
import cProfile

Count = 0
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.disable_eager_execution()

class DataDistribution(object):  # 真实数据分布
    def __init__(self):  # 只执行一次
        self.mu = 4  # 线　的参数  高斯分布相关

        self.sigma = 0.5

    def sample(self, N):  # 每次迭代都运行
        samples = np.random.normal(self.mu, self.sigma, N)  # 正态分布
        samples.sort()  # 画图   原来真实分布

        return samples


class GeneratorDistribution(object):  # 随机初始化分布,当成G输入
    def __init__(self, range):
        self.range = range

    def sample(self, N):  # 迭代次数的两倍

        return np.linspace(-self.range, self.range, N) + \
               np.random.random(N) * 0.01


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def linear(input, output_dim, scope=None, stddev=1.0):  # 100次迭代 运行了14次
    norm = tf.random_normal_initializer(stddev=stddev)  # 随机 w
    const = tf.constant_initializer(0.0)  # b

    with tf.compat.v1.variable_scope(scope or 'linear'):
        w = tf.compat.v1.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)  # 随机w
        b = tf.compat.v1.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')

    return h1


def discriminator(input, h_dim):  # #初始化 w b
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))

    return h3


# @static_vars(counter = 0)
def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95

    num_decay_steps = 150  # 每迭代150次 做一次学习率的衰减
    batch = tf.Variable(0)

    learning_rate = tf.compat.v1.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    # optimizer.counter += 1

    # print("counter:", optimizer.counter)
    return optimizer


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data  # DataDistribution
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4  # 用神经网络模型当G D 网络时,隐层神经元个数
        self.count = 0
        self.learning_rate = 0.03  # 学习率

        self._create_model()

    def _create_model(self):  # 1/100次

        with tf.compat.v1.variable_scope('D_pre'):  # 先训练D的初始化
            self.pre_input = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)  # 初始化 w b
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.compat.v1.variable_scope('Gen'):
            self.z = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.compat.v1.variable_scope('Disc') as scope:
            self.x = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size)  # 构造D1网络,真实数据当做输入
            scope.reuse_variables()  # 重用
            self.D2 = discriminator(self.G, self.mlp_hidden_size)  # 构造D2网络,G数据当做输入

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-tf.compat.v1.log(self.D1) - tf.compat.v1.log(1 - self.D2))  # 希望self.D1 向1...self.D2 向0   计算结果是均值
        self.loss_g = tf.reduce_mean(-tf.compat.v1.log(self.D2))  # 1

        self.d_pre_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):  # 1

        with tf.compat.v1.Session() as session:
            tf.compat.v1.global_variables_initializer().run()

            # pretraining discriminator
            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)  # 高斯
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)
            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightsD[i]))

            for step in range(self.num_steps):
                # update discriminator
                x = self.data.sample(self.batch_size)  # 原始的分布图  x是随机的
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {  # 自动调网络?????
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator
                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))
                if step % 100 == 0 or step == 0 or step == self.num_steps - 1:
                    self._plot_distributions(session)

    def _samples(self, session, num_points=10000, num_bins=100):  # 2/100次

        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)
        return pd, pg

    def _plot_distributions(self, session):  # 2/100

        pd, pg = self._samples(session)
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()


def main(args):
    model = GAN(
        DataDistribution(),  # 真实数据分布   只执行一下init
        GeneratorDistribution(range=8),  # 随机初始化分布 当成G 输入
        args.num_steps,  # 迭代次数
        args.batch_size,  # 一次迭代 个
        args.log_every,  # 隔多少次打印loss
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,  # 1200
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
