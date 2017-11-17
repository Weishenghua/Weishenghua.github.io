---
layout: post
title:  "Linux下 安装YOLO"
date:   2017-11-14
categories: linux, object detection
tags: linux, object detection
---

YOLO 是一个实时的目标检测框架。 最近由于实验室需要进行实验，所以要在服务器上对于YOLO进行安装。YOLO的网址为 <https://pjreddie.com/darknet/yolo/>.

You only look once (YOLO) is a state-of-the-art, real-time object detection system. On a Titan X it processes images at 40-90 FPS and has a mAP on VOC 2007 of 78.6% and a mAP of 48.1% on COCO test-dev.

下面介绍安装YOLO的具体步骤。


### 1.安装Darknet 
Darknet 是一个开源的神经网络的框架，采用C和CUDA编写。简单方便并且支持CPU和GPU。
在使用过程中可能会需要Opencv和Cuda，所以需要首先安装这两个东西。
#### 1.1 安装cuda
按照要求安装在 /usr/local 目录下。如果在多用户的Linux下，可以在自己的用户目录下配置相应的cuda。
#### 1.2 安装opencv
按照要求正常的安装opencv 在/usr/local下。
#### 1.3 安装darknet
opencv 和 CUDA 可以帮助darknet更好的显示载入图片和加速神经网络的速度。
在安装好上述两者后，开始进行darknet的安装。

首先从git上克隆darknet文件，其次进行make。
<pre><code>
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
</code></pre>

在make之前需要进行GPU和opencv的配置，在```/darknet``` 目录下的 ```makefile``` 文件前几行更改为：
<pre><code>
GPU=1
CUDNN=1
OPENCV=1
</code></pre>

如果使用自己的CUDA路径的话，在这里将CUDA路径改为自己的路径：
<pre><code>
ifeq（$(GPU),1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -Lcudart -lcublas -lcurand
</code></pre>

### 1.2 配置YOLO
之前已经安装了Darknet，其中包含了基本的神经网络的库。在Darknet中，包含了```cfg``` 文件夹，其中含有数据集的配置文件以及网络结构的配置文件。首先可以尝试使用作者训练好的YOLO权重。
#### 1.2.1 使用预训练的YOLO模型
<pre><code>
wget https://pjreddie.com/media/files/yolo.weights
</code></pre>
然后就可以直接进行测试了！！！！
<pre><code>
./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg
</code></pre>
会显出出网络的结构以及预测的几个类别的结果。
并且会显示出图像。
还可以同时测试多张图像，或者修改显示结果的权重。作者同样提供了基于网络摄像头的实时监测API以及Tiny-YOLO.
####1.2.2 使用自己的数据进行训练
作者的网站上提供了下载VOC数据集并且在VOC数据集上进行训练的教程，这里只介绍使用自己的数据进行训练的过程。
对于Darknet 这个框架，训练数据一共分为三部分，一部分是jpg图片，一部分是每一个jpg图片的标记txt文件，最后是划分数据集的train.txt,val.txt和test.txt。其中图片标记的文件格式如下：

```
<object-class> <x> <y> <width> <height>
```

划分数据集的文件每一行包含了图片文件的路径。

制作好数据集后，需要进行一点配置, 主要需要三个文件，一个是```/cfg```文件夹下的 ```.data```配置文件，主要包含的信息是数据集的信息以及类别信息和数据及每一类的名称(```voc.names```)。需要根据自己的情况进行修改，并且穿件backup文件夹用于存储训练权重的备份。
<pre><code>
classes= 20
train  = path-to-voc/train.txt
valid  = path-to-voc/test.txt
names = data/voc.names
backup = backup
</code></pre>

第二个文件```voc.names``` 按行存储每一类的名称就好。

最后一个是网络的配置文件```cfg/yolo-voc.cfg```,训练的时候按照自己的需要进行修改。

由于上述的文件都是作者已经提供了的，当我们使用自己的数据集时最好自己复制一下重命名已有的文件，在自己的配置文件上进行修改。

最后一步，训练！！！！
<pre><code>
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
</code></pre>

``` darknet19_448.conv.23 ``` 代表使用预训练的权重，不想使用的话直接不写这部分即可。


### 总结

YOLO的作者自己实现了神经网络框架，容易配置且方便使用，后面还要继续搞明白一些细节便于更好的训练自己的数据。