# Python的包管理工具: `pip`

由于python是模块化的开发，因此能够能够利用其他人写的现成的包来快速的完成特定的任务。为了加快包的安装，python有很多包管理的工具，其中`pip`是目前使用最多的包管理工具。

## 1. 安装pip
在ubuntu系统可以直接安装python-pip

```
# Python 3的pip （建议安装Python3）
sudo apt-get install python3-pip

# Python 2的pip
sudo apt-get install python3-pip
```

Upgrade pip
```
sudo pip3 install --upgrade pip
```

安装之后，可以输入`pip`查看简要的使用说明。**需要注意的是，通过系统安装的pip，在使用pip安装包的时候，需要用sudo来执行。**


## 2. pip的命令

### 2.1 查找一个给定名字的package
```
pip search numpy
```
会找到很多跟numpy有关联的包，可以拷贝每一行最前面的那个包名字，通过安装命令去安装。


### 2.2 安装一个给定的package
```
$ pip install numpy
```
安装`numpy`这个包，同时它的依赖也自动安装到系统。

使用一个给定的URL安装包
```
$ pip -f URL install PACKAGE    # 从指定URL下载安装包
```


### 2.3 升级一个包
```
$ pip -U install PACKAGE        # 升级包
```

### 2.4 列出当前系统中已经安装的包
```
$ pip list
```

查看一个安装好的包的信息
```
$ pip show numpy
```


## 3. 设置pip的镜像
但是由于直接使用pip去访问国外的网站慢，所以需要设置好pip的镜像，从而加快包的安装。目前国内有很多pip包镜像，选择其中一个就可以加快很多安装速度

```
pip config set global.index-url 'https://mirrors.ustc.edu.cn/pypi/web/simple'
```