---
layout: post
title:  "多用户Linux下配置python,pip,anaconda,jupyter notebook"
date:   2017-10-11
categories: python,linux  
tags: python, linux
---
最近在实验室的服务器上希望配置python 和 tensorflow 来进行一些实验，但是在配置的过程中遇到了一些小问题，在查阅资料后基本了解了linux自带python，pip， anaconda， conda 等等之间的关系。记录下来，备不时之需。

### Linux 自带 python:
Linux(Ubuntu)自带python，包括python2.7 和 python3.5，存储在/usr/bin 中， 系统中的默认**python** 和 **python3** 指令会调用默认的版本。
### pip:
The PyPA recommended tool for installing Python packages. 用于管理python安装包。
### conda:
同样是用于管理安装包的工具，但是不仅限于python。
### anaconda:
是一个集成好的python安装包，包括了很多科学计算需要的包， 如 numpy， scipy等
### ~/.bashrc
这个文件中的一些指令决定了环境变量。
可以在这个文件中修cuda 的指向。
cuda中包括不同版本的cudnn。在大家公用一台机器时可能配置不同，这时可以配置自己的 ```bashrc```文件指向自己的cuda路径。


## 安装anaconda 并且更换系统中的默认python
1）安装anaconda
直接 ```bash Anaconda2-4.4.0-Linux-x86_64.sh```（一路enter、yes、ok之类的）</br>
可以考虑```source ~/.bashrc</br>```
注意，这里可以```which python```一下，如果是```/home/username/anaconda2/bin/python```，就ok</br>。
如果是```/usr/bin/python```，则说明python指向不对；此时看下anaconda安装结束之前的这个问题是不是选择yes：</br>
```creating default environment...```</br>
```installation finished.```</br>
```Do you wish the installer to prepend the Anaconda2 install location to PATH in``` 
```your /home/wshh/.bashrc ? [yes|no]```</br>
```[no] >>> yes```

2）然后```pip install tensorflow```

3）然后```pip install keras```

## 如何在系统自带python和Anaconda切换

Alias 别名

```alias py27="/usr/bin/python2.7" ```
```alias pyana="/home/myname/anaconda2/bin/python2.7" ```
