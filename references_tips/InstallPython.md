# 安装Python环境

由于Python的库比较多，并且依赖关系比较复杂，所以请仔细阅读下面的说明，并按下面的说明来操作，减少问题出现的可能。 **但是所列的安装方法说明里有较多的细节，也许和你的系统并不适配，所以会遇到问题。如果遇到问题请通过搜索引擎去查找解决的办法**，通过这个方式锻炼自己解决问题的能力。

可以参考后面所列的`1.Winodws`或者`2.Linux`章节所列的将Python环境安装到计算机里。


## 1. Windows下安装

由于Anaconda集成了大部分的python包，因此能够很方便的开始使用。由于网络下载速度较慢，因此推荐使用镜像来提高下载的速度。镜像的使用方法可以参考：[Anaconda镜像的说明文档](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda)

1. 在这里找到适合自己的安装文件，然后下载
   https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

例如： https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2020.11-Windows-x86_64.exe

2. 按照说明，把Anaconda安装好。



## 2. Linux下安装
在网站下载最新的conda安装文件，例如

```
wget https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2020.11-Linux-x86_64.sh
```

然后运行
```
bash ./Anaconda3-2020.11-Linux-x86_64.sh
```

按照提示完成安装（记得需要自动加入环境变量的设置），**然后关闭终端，再打开终端**


## 3. 设置软件源
### 3.1 设置conda软件源 

参考这里的[conda安装和软件源设置说明](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

或者其他源
```
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/
```


### 3.2 设置PIP源

```
pip config set global.index-url 'https://mirrors.ustc.edu.cn/pypi/web/simple'
```

## 4. 安装常用软件
打开`conda`的命令行程序，输入下面的命令
```
conda install jupyter
conda install scipy
conda install numpy
conda install sympy
conda install matplotlib
conda install pandas
conda install scikit-learn
```

## 5. 安装PyTorch

```
conda install pytorch -c pytorch 
pip3 install torchvision
```




## 6. Conda使用技巧

### 6.1 Conda创建自己的环境
```
conda create -n xueshaocheng_pytorch
```

### 6.2 Conda怎么激活自己的环境
```
conda activate xueshaocheng_pytorch
```

### 6.3 Conda常用命令
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

## 7. [Python技巧](python/)

- [pip的安装、使用等](python/pip.md)
- [virtualenv的安装、使用](python/virtualenv.md)
- [virtualenv便捷管理工具：virtualenv_wrapper](python/virtualenv_wrapper.md)

* Anaconda使用技巧