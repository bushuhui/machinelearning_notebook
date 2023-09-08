# CIFAR-10 Image Classification in PyTorch

This repo mainly focuses on the standard pipeline of one popular computer vision task 'image classification'. [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) is one popular toy image classification dataset, which consists of 
60,000 tiny images that are 32 pixels high and wide.Each image is labeled with one of 10 classes. These 60,000 images are partitioned into a training set of 50,000 images and a test set of 10,000 images.

**The image classification pipeline** Normally, the complete pipeline for an image classification which belongs to supervised learning field can be formulated as follow:
* **Input**: Our input consists of a set of N images, each labeled with one of K different classes. We refer to this data as the *training set*.
* **Learning**: Our task is to use the training set to learn what every one of the classes looks like. We refer to this step as *training a classifier*, or *learning a model*.
* **Evaluation**:  In the end, we evaluate the quality of the classifier by asking it to predict labels for a new set of images that it has never seen before. We will then compare the true labels of these images to the ones predicted by the classifier. Intuitively, we’re hoping that a lot of the predictions match up with the true answers (which we call the *ground truth*).

## Installing and setting up your virtual environment

Firstly, we create a virtualenv for this project (we assume that you haven't installed the pip on your computer). 
In many cases, Python 3 may not be installed in your OS. Installing Python 3 is highly dependent on the operating system. For this process, you will need root access. If you don’t please contact your system administrator. Also, depending on your OS, you might even have Python 3 as default. For those systems, you probably know what you are doing, so we won’t go into the details.

**Linux**: If you are using a Linux distribution, use your package manager to install Python 3 and pip. For example, on Ubuntu, it should be as simple as typing in the terminal:
```
sudo apt-get install python3 python3-pip
```
**Mac OSX**: Use Homebrew. Install Homebrew by visiting brew.sh, and type in the terminal
```
brew install python3
```
**Windows**: Please install a Linux Virtual Machine using Virtual Box, or use a Linux server. Then follow Linux instructions.

**Python 3.7 compatibility issues**: Python 3.7 is not compatible with many libraries. It is highly recommended that you stick to Python 3.6. For installing Python 3.6, please refer to the package manager’s documentations.

After installing pip, you can easily install your virtual environment by opening a terminal and typing:
```
pip3 install --user virtualenv
```
Here, we have the --user flag as we are installing this for the current user. This way we won’t need root privileges.

Then, you can set up your virtualenv:
```
virtualenv --python=$(which python3) ∼/my_venvs/cifar_10
source ~/my_venvs/cifar_10/activate
```
Note: when create a virtualenv, you need to specify a directory for your virtual environments. For example, it could be ∼/my_venvs/cifar_10. We will use this directory for the time being, for easy explanation, but any directory is fine.
To leave the virtual environment, type deactivate. In most cases, you will see something on the left of your command prompt, showing that you are inside a virtual environment. Once inside a virtual environment, you should now be able to install the exact environment that you will be evaluated through:
```
pip3 install -r requirements.txt
```

## Preprocessing the data
Even though there is a built-in data loader in PyTorch for [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html), we choose to write the data loader by ourself for better understanding, which is stored in the `utils` folder.

## Implementation ##

In this repo, we also offer two ways of convolution layers, one is used torch library and another is customize by ourself. In addition, the basic architecture is based on ResNet.Implement the creation of convolution layers according to the comments in the code. The residual network can be invoked with the following command line:
```
python solution.py --log dir logs/custom --save dir saves/custom --conv custom --data_dir (your data path) --mode test --resume False
```
Note: please change the *data_dir* to your downloaded CIFAR-10 absolute path, otherwise it will cause errors. All parameters setting can be seen in `config.py` file.

With the provided models, you should obtain the following test performance. These performances are with all other parameters set to default, except for conv.

| Configuration | Test Accuracy (%)|
| ------------- | ------------- |
| --conv torch  | 49.79  |
| --conv custom | 62.68  |

## Going beyond ##

Use the knowledge you’ve acquired to enhance the performance of your model. You may change your architectural choice, and/or apply any other type of method that you would like to.
Note that this would probably take some training time. 

**Hint**: It is possible to achieve this accuracy simply by changing the hyper-parameters. However, you may also do whatever you want to do instead. 
For example, it can use the learing_rate scheduler technique which is written in the `model.py`. In the `save` folder, it offered the bestmodel weights and you can also test it.

All parameter changes can simply modify based on the `config.py` file. If you have no idea for *argparse*, please refer to the [Argparse Tutorial](https://docs.python.org/3.6/howto/argparse.html). The invoking
command line is similar to the reproduction of the baseline test performance.

## Problems ##

If you have encounter any issue, please report it! Good luck :smile: :smile: :smile: