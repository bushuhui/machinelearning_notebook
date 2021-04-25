# 安装Python环境

由于Python的库比较多，并且依赖关系比较复杂，所以请仔细阅读下面的说明，并按下面的说明来操作，减少问题出现的可能。 **但是所列的安装方法说明里有较多的细节，也许和你的系统并不适配，所以会遇到问题。如果遇到问题请通过搜索引擎去查找解决的办法**，通过这个方式锻炼自己解决问题的能力。

可以参考后面所列的`1.Winodws`或者`2.Linux`章节所列的将Python环境安装到计算机里。如果想一次性把所有的所需要的软件都安装到机器上，可以在本项目的根目录下执行下面的命令，需要Python 3.5以上的版本，如果出现问题，则可以参考`requirements.txt`里面所列的软件包名字，手动一个一个安装。
```
pip install -r requirements.txt
```


## 1. Windows

### 1.1 安装Anaconda

由于Anaconda集成了大部分的python包，因此能够很方便的开始使用。由于网络下载速度较慢，因此推荐使用镜像来提高下载的速度。镜像的使用方法可以参考：[Anaconda镜像的说明文档](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda)

1. 在这里找到适合自己的安装文件，然后下载
   https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

2. 按照说明，把Anaconda安装好。

3. 设置软件源 https://mirror.tuna.tsinghua.edu.cn/help/anaconda/

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

### 1.2 安装PyTorch
```
conda install pytorch -c pytorch 
pip3 install torchvision
```




## 2. Linux

### 2.1 安装pip
```
sudo apt-get install python3-pip
```



### 2.2 设置PIP源

```
pip config set global.index-url 'https://mirrors.ustc.edu.cn/pypi/web/simple'
```



### 2.3 安装常用的包

```
pip install -r requirements.txt
```

或者手动安装
```
sudo pip install scipy
sudo pip install scikit-learn
sudo pip install numpy
sudo pip install matplotlib
sudo pip install pandas
sudo pip install ipython
sudo pip install jupyter
```



### 2.4 安装pyTorch

到[pytorch 官网](https://pytorch.org)，根据自己的操作系统、CUDA版本，选择合适的安装命令。

例如Linux, Python3.5, CUDA 9.0：
```
pip3 install torch torchvision
```



## 3. Conda使用技巧

### 3.1 Conda创建自己的环境
```
conda create -n xueshaocheng_pytorch
```

### 3.2 Conda怎么激活自己的环境
```
conda activate xueshaocheng_pytorch
```

### 3.3 Conda常用命令
```
# 帮助命令
conda -h
conda help

# 配置频道(已有)
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/

# 退出当前环境
conda deactivate

# 查看基本信息
conda info
conda info -h

# 查看当前存在环境
conda env list
conda info --envs

# 删除环境
conda remove -n yourname --all
```

## 4. [Python技巧](python/)

- [pip的安装、使用等](python/pip.md)
- [virtualenv的安装、使用](python/virtualenv.md)
- [virtualenv便捷管理工具：virtualenv_wrapper](python/virtualenv_wrapper.md)

* Anaconda使用技巧