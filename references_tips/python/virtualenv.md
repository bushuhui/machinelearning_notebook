# virtualenv manual


## 1. Install
virtualenv 是一个创建隔绝的Python环境的工具。virtualenv创建一个包含所有必要的可执行文件的文件夹，用来使用Python工程所需的包。
```
pip install virtualenv
```

如果当前pip是python2的话，则后续默认创建的虚拟环境就是python2；否则是python3的


## 2. 创建虚拟环境

创建一个虚拟环境
```
$ mkdir -p ~/virtualenv; cd ~/virtualenv

$ virtualenv venv    # venv 是虚拟环境的目录名
```

virtualenv venv 将会在当前的目录中创建一个文件夹，包含了Python可执行文件，以及 pip 库的一份拷贝，这样就能安装其他包了。虚拟环境的名字（此例中是 venv ）可以是任意的；若省略名字将会把文件均放在当前目录。

在任何你运行命令的目录中，这会创建Python的拷贝，并将之放在叫做 venv 的文件中。

你可以选择使用一个Python解释器：
```
$ virtualenv -p /usr/bin/python2.7 venv　　　　# -p参数指定Python解释器程序路径
```


## 3. 使用虚拟环境

要开始使用虚拟环境，其需要被激活：
```
$ source ~/virtualenv/venv/bin/activate　　　
```

从现在起，任何你使用pip安装的包将会放在 venv 文件夹中，与全局安装的Python隔绝开。

像平常一样安装包，比如：
```
$ pip install requests
```


## 4. 如果你在虚拟环境中暂时完成了工作，则可以停用它：
```
$ . venv/bin/deactivate
```
这将会回到系统默认的Python解释器，包括已安装的库也会回到默认的。


## 5. 删除一个虚拟环境
要删除一个虚拟环境，只需删除它的文件夹。（执行 rm -rf venv ）。


这里virtualenv 有些不便，因为virtual的启动、停止脚本都在特定文件夹，可能一段时间后，你可能会有很多个虚拟环境散落在系统各处，你可能忘记它们的名字或者位置。

