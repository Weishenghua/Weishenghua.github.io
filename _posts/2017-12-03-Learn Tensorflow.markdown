---
layout: post
title:  "Learn tensorflow"
date:   2017-12-03
categories: python tensoflow
tags: python tensoflow
---
# 简介
之前对于tensorflow 进行了简单的学习，了解了其基本内容，后面就要对于其几个例子进行学习。

## Minst for ML beginners
这个例子对于机器学习的初学者设计。 Minst是一个手写数字数据集，包含0~9的数字，代码都在Tensorflow提供的minst_softmax.py代码中。

Mist 数据集包含55000个训练样本， 10000个测试样本和5000个验证样本。
image 为 x， label 为 y。每一个图像是一个28 x 28 = 784个像素点组成的。 相当于每一张图像是一个784-d的向量。 训练集为55000x784的矩阵，而标签哪位55000x10的矩阵(10类）。这个简单的例子采用softmax回归。公式为 y = softmax(wx+b).

要实现这个简单的功能，实现需要导入Tensorflow，然后定义输入的x为一个plcaeholder。维度为[None,784].
W 和 b 都为tensorflow中的变量。
y为 x，W和b这个线性模型的计算结果加上softmax函数。

<pre><code>
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
</code></pre>

在完成模型的初始化，就要进行训练，要训练就需要ground truth， 仍然使用placeholder 表示：
<pre><code>
y_ = tf.placeholder(tf.float32, [None, 10])
</code></pre>

然后需要定义交叉熵损失函数:
<pre><code>
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
</code></pre>

最后使用梯度下降法来进行模型的优化,初始的学习率设置为0.5:
<pre><code>
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
</code></pre>

最终使用Session来计算梯度和优化模型:
<pre><code>
tf.global_variables_initializer().run()
</code></pre>

初始化模型参数（W，b）：
<pre><code>
tf.global_variables_initializer().run()
</code></pre>

开始进行训练:
<pre><code>
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
</code></pre>

评估模型，最终还是要通过sess来计算准确率：
<pre><code>
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
</code></pre>

这样的简单模型结果可以达到92%!


## Deep MNIST for Experts

之前尝试使用简单的线性模型来进行手写字符识别的任务，现在通过卷积神经网络————一种更复杂但是有效的模型来进行手写字符识别. 其内容包含在minst_deep.py中。

首先通过Tensorflow提供的API来读取数据。然后导入Tensorflow 和Intesractive Session。 Tensorflow依赖一个强大的C++后端来进行使计算，而和后端之间的连接（Connection）就是Session。通常程序的模式就是（1）创建一个计算图；(2)在Session中启动进行训练计算等。

目前在这里使用Interactive Session，实际上为代码提供了更加灵活的实现方式。可以交错的创建graph和运行graph。这在使用ipython时就很方便。如果不使用它的话，就必须首先创建好整个graph，再在Session中launch它。
<pre><code>
import tensorflow as tf
sess = tf.InteractiveSession()
</code></pre>

在进行一些昂贵的操作的时候（矩阵乘法），我们常常通过使用其他的库（Numpy)来在外部进行操作，使用更高效的语言（C++），但是在使用高效的库和切换回python之间有很大的overhead。这种Overhead在你希望分布式地在GPU上进行计算时尤为明显，有很大的开销都在传输数据上。

Tensorflow也有高效庞大的外部库，但是为了避免overhead，它不会只在python外部运行单个昂贵的操作，而是允许我们定义一个图，它完全在python外部运行。这里的python代码的作用就是构造外部的计算图，并且声明图的哪些部分是要被运行的即可。

之前已经构造过简单的模型，下面开始构造一个多层的卷积神经网络模型。在Minst上应该要有比简单线性模型更高的识别准确率。

权重初始化，使用的是RELU，不能都是0， 那么都是dead relu；W 要有随机性打破对称性；太大饱和，太小梯度弥散：
<pre><code>
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
</code></pre>

定义卷积层：
<pre><code>
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
</code></pre>

网络结构，注意这里卷积模板的参数，前两个表示wxh的滤波器大小，第三个表示滤波器的depth（channnel），最后一个表示输出的滤波器个数； x_image为一个4维的tensor，中间两位为图像的长宽，最后的一维为图像的channels：
<pre><code>
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

</code></pre>


最后进行训练：
<pre><code>
ross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
</code></pre>
