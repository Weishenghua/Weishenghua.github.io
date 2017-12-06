---
layout: post
title:  "Get Started with tensorflow"
date:   2017-11-30
categories: linux, python tensoflow
tags: python tensoflow
---
# 简介
Tensorflow 是一个开源的软件包，使用data flow graph 来进行数值计算。 计算图中的节点表示数学操作，图中的边表示多维数组（张量， tensor）。它可以方便的被用于CPU，GPU，Server， mobile devices。 现在主要支持的就是python 和 C++.

# Data flow graph
数据流图描述了一个由节点和边构成的有向图的数学计算。 节点常被用于实现数学计算，同时也可以表示输入数据和输出结果或者读写永久变量的端点（end point）。这些数据边包含了动态大小的多维数组或者张量。张量的flow穿过这个graph，构成了它的名字， Tensorflow。节点被分配到计算单元，它们被异步的计算；当不同节点的输入张量都可用时，它们会并行的被计算。


# Getting Started with tensorflow
基础：python 编程以及数组和机器学习的概念。

Tensorflow 有不同层次的API， 最底层的是Tensorflow Core. 高层的API基于CORE，并且减少重复性的工作。 底层的API提供对于代码更好的操作性。

首先开始进行Core 的一些简单的介绍，后面用更高级的API实现同样的功能

# Tensors
Tensor 就是一个高维数组，是Tensorflow 的基础， tensor 的rank就是它的维度。

<pre><code>
3 # a rank 0 tensor; a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
</code></pre>

## Tensor Core 教程
### 导入tensorflow
<pre><code>
import tensorflow as tf
</code></pre>
### 计算图 Computational Graph
TensorFlow core 代码大致分为两步

1. 构造计算图
2. 运行计算图

计算图就是将一系列的操作赋予图的节点。每一个节点输入一个或者多个tensor，并且输出1个tensor。下面构建常量node（无输入，有常数输出）。

<pre><code>
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
</code></pre>

输出：

<pre><code>
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
</code></pre>

没有按照预期的输出3和4， 这是由于计算图必须放在session中。Session封装了Tensorflow运行时的控制和状态。需要创建Session并且run一下。
<pre><code>
sess = tf.Session()
print(sess.run([node1, node2]))
</code></pre>

成功的输出了
<pre><code>
[3.0, 4.0]
</code></pre>

我们可以通过结合Tesor节点和操作（操作也是节点）来进行复杂的计算。两者的加法：
<pre><code>
from __future__ import print_function
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
</code></pre>

输出为
<pre><code>
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
</code></pre>

Tensorboard可以绘制计算图。

到目前为止使用的都是常量而不是动态的量。通过placeholder可以参数化计算图并且接受外部数据。
<pre><code>
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
</code></pre>

使用session运行时，run函数输入两部分，一部分是节点，另一部分是输入的数据
<pre><code>
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
</code></pre>

输出的结果如下：
<pre><code>
7.5
[ 3.  7.]
</code></pre>
如果不输入数据直接去输出的话会报错。

添加操作使得计算图更加复杂:
<pre><code>
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
</code></pre>

输出：
<pre><code>
22.5
</code></pre>

机器学习中希望模型能接收任意的输入，要使模型可训练的话，使用 Variables。 它可以被初始化并且通过训练更改数值。
<pre><code>
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
</code></pre>

Constant常量在构造的时候（运行 constant的时候） 就已经初始化了， 并且数值会固定下来， 永远不会变， 而 Variable 不是这样的， 希望初始化的话必须这样：
<pre><code>
init = tf.global_variables_initializer()
sess.run(init)
</code></pre>
其中init是一个用于初始化所有全局变量的子图的句柄（handle）。
X 为一个placeholder， 所以对于它属入数据，如下：
<pre><code>
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
</code></pre>
输出这样的结果：
<pre><code>
[ 0.          0.30000001  0.60000002  0.90000004]
</code></pre>

我们构建了一个模型，但是我们不知道好坏。为了评估模型，我们需要一个y placeholder和loss function。 代码如下：
<pre><code>
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
</code></pre>
这段代码中首先构建了一个y的placeholder， 然后将其和Linear_model作差，并且计算平方和平方和。 输出的误差为：
<pre><code>
23.66
</code></pre>
我们还可以通过assign操作对于Varible重新赋值，对于W和b进行赋值：
<pre><code>
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
</code></pre>
loss 变化为
<pre><code>
0.0
</code></pre>
在这次测试中我们直接修改W,b的值，但在实际中我们需要训练来更新参数.

## tf.train API
Tensorflow 提供了一些optimizer来更新参数，其中最常见的就是梯度下降（gradient descent）。根据每一个Variable对于损失的导数来更新他们：
我们还可以通过assign操作对于Varible重新赋值，对于W和b进行赋值：
<pre><code>
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b])
</code></pre>

###完整的代码是：
<pre><code>
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
</code></pre>

### tf.esimater
这是一个high-level的 Tensorflow的库， 简化了机器学习流程，它可以运行训练和验证的循环并且可以管理数据。它还包含了很多常见模型：

基础使用：
<pre><code>
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import tensorflow as tf

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
</code></pre>