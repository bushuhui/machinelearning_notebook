
# config.py ---
#
# Filename: config.py
# Description: Based on argparse usage from
#              https://github.com/carpedm20/DCGAN-tensorflow
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jun 26 11:06:51 2017 (+0200)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#

# Code:

import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--data_dir", type=str,
                       default="./data/cifar-10-batches-py",
                       help="Directory with CIFAR10 data")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-1,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=int,
                       default=25,
                       help="Number of epochs to train")

train_arg.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

train_arg.add_argument("--resume", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")
# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--loss_type", type=str,
                       default="cross_entropy",
                       choices=["cross_entropy", "svm"],
                       help="Type of data loss to be used")

model_arg.add_argument("--normalize", type=str2bool,
                       default=True,
                       help="Whether to normalize with mean/std or not")

model_arg.add_argument("--l2_reg", type=float,
                       default=1e-4,
                       help="L2 Regularization strength")

model_arg.add_argument("--num_unit", type=int,
                       default=128,
                       help="Number of neurons in the hidden layer")

model_arg.add_argument("--num_hidden", type=int,
                       default=3,
                       help="Number of hidden layers")

model_arg.add_argument("--nchannel_base", type=int,
                       default=8,
                       help="Base number of channels")

model_arg.add_argument("--ksize", type=int,
                       default=3,
                       help="Size of the convolution kernel")

model_arg.add_argument("--num_conv_outer", type=int,
                       default=3,
                       help="Number of outer blocks")

model_arg.add_argument("--num_conv_inner", type=int,
                       default=1,
                       help="Number of convolution in each block")

model_arg.add_argument("--num_class", type=int,
                       default=10,
                       help="Number of classes in the dataset")

model_arg.add_argument("--conv2d", type=str,
                       default="torch",
                       help="Convolution type")

model_arg.add_argument("--pool2d", type=str,
                       default="MaxPool2d",
                       help="Pooling type")

model_arg.add_argument("--activation", type=str,
                       default="ReLU",
                       help="Activation type")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

