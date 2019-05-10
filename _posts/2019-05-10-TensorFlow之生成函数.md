---
layout:     post
title:      "TensorFlow之生成函数"
subtitle:   "今天来说一下TensorFlow中的生成函数，为什么需要生成函数呢?"
date:       2019-05-10
author:     "木夏"
header-img: "img/post-bg-css.jpg"
catalog: true
mathjax: true

tags:
    - TensorFlow
---



>`当时她还很年轻，不知道命运所赠予的礼物，早已在暗中标好了价格。——茨威格《断头王后》`

>说一下这句话的来源吧，因为实在太喜欢这句了。从未刻意记住，但是不曾忘却，就像是在茫茫人海中的飘然一眼，从此遗落万年。

>路易十六的老婆玛丽·安托瓦内特，14岁的时候就成为法国的太子妃，18岁成为法国王后。丈夫很爱她，由着她的性子建宫殿，办宴会，夜夜笙歌，以至于玛丽·安托瓦内特的亲哥哥从奥地利专程来法国规劝自己的亲妹妹，对她说你现在是法兰西王后，你能不能每天读一小时书，这并不难。玛丽对哥哥说：我不喜欢读书，我喜欢享受生活。
20年后，玛丽·安托瓦内特上了断头台，被称为断头王后。

## Begin
今天来说一下TensorFlow中的生成函数，为什么需要生成函数呢?因为我们的模型训练需要输入数据，如果手中没有真实数据，而又想训练模型怎么办，那就需要我们自己生成数据。那么这样来说，我们就需要到生成函数了。TensorFlow中的生成函数，可以很方便地生成我们需要的数据。

## 1.随机数生成函数
利用以下函数可以生成随机数。
### 1.1生成正太分布随机数
生成正态分布随机数的函数为：`tf.random_normal()`
该函数有以下参数：`stddev:标准差`、`mean：平均值`、`形状`。
用python生成一个正态分布矩阵$W_1$：
```python
#我们生成一个3行4列的正态分布矩阵，其中标准差stddev=3，平均值mean=0，随机种子为2，指定随机种子是便于复现
W_1 = tf.Variable(tf.random_normal( [3,4],stddev=3, mean=0, seed=2))
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出生成的矩阵W_1
    print(sess.run(W_1))
```
输出正态分布矩阵$W_1$如下：
$$W_1=\begin{bmatrix}
    [-2.5743325,-0.58986896,0.41685134,-3.6638303 ]\\
    [-1.2102386,-3.4362123 , 1.9194721, -4.7298355 ]\\
    [-3.078032 , 1.885525  , -2.4846714 ,  1.0400314]
    \end{bmatrix}$$

### 1.2生成**去掉过大偏离点**的正态分布随机数
生成正态分布，但是如果随机出来的值偏离平均值超过2个标准差，那么这个数将会被重新再次随机。
该函数为：`tf.truncated_normal()`
其中参数有：`stddev:标准差`、`mean：平均值`、`形状`
用python生成一个**去掉过大偏离点**的正态分布矩阵$W_2$：
```python
#我们生成一个3行4列的去掉过大偏离点的正态分布矩阵，其中标准差stddev=3，平均值mean=0，随机种子为2，指定随机种子是便于复现
W_2 = tf.Variable(tf.truncated_normal([3,4],stddev=3, mean=0, seed=2))
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出生成的矩阵W_2
    print(sess.run(W_2))
```
输出去掉过大偏离点的正态分布矩阵$W_2$如下：
$$W_2=\begin{bmatrix}
    [-2.5743325 , -0.58986896 , 0.41685134 ,-3.6638303 ]\\
    [-0.6237943 , -5.334839  , -1.3404311 , -0.33711153]\\
    [-3.2932882  , 1.2972201 , -0.12828581 , 0.8206395 ]\\
    \end{bmatrix}$$

### 1.3生成均匀分布的随机数
生成均匀分布的函数为:`tf.random_uniform()`
参数主要有：`shape：形状`、`minval:最小值`、`maxval：最大值`、`dtype：类型`
用python生成一个均匀分布矩阵$W_3$：
```python
#生成一个2行3列的均匀分布矩阵，从[minval,maxval)中随机采样，下面即为从[0,1)中采样，类型为tf.float32，随机种子为1，指定随机种子是便于复现
W_3 = tf.Variable(tf.random_uniform(shape=[2,3],minval=0,maxval=1,dtype=tf.float32,seed=1))
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出生成的矩阵W_3
    print(sess.run(W_3))
```
输出均匀分布矩阵$W_3$如下：
$$W_3=\begin{bmatrix}
    [0.2390374 ， 0.92039955 ，0.05051243]\\
    [0.49574447 ，0.8355223 ， 0.02647042]\\
    \end{bmatrix}$$


## 2.常数生成函数
除了生成随机数之外，还可以生成常数。
### 2.1生成全0的矩阵
函数：`tf.zeros()`
参数：`shape:形状`、`dtyp:类型`
用python生成一个生成全0的矩阵$C_0$：
```python
#定义一个全0矩阵，形状为2行3列，类型为tf.int32
C_0 = tf.Variable(tf.zeros(shape=[2,3],dtype=tf.int32))
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出生成的矩阵C_0
    print(sess.run(C_0))
```
输出全0矩阵$C_0$如下（2行3列）：
$$C_0=\begin{bmatrix}
    [0， 0 ，0]\\
    [0， 0 ，0]\\
    \end{bmatrix}$$
    
### 2.2生成全1的矩阵
函数：`tf.ones()`
参数：`shape:形状`、`dtyp:类型`
用python生成一个生成全1的矩阵$C_1$：
```python
#定义一个全1矩阵，形状为3行2列，类型为tf.float32
C_1 = tf.Variable(tf.ones(shape=[3,2],dtype=tf.float32))
#用会话进行节点计算
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出生成的矩阵C_1
    print(sess.run(C_1))
```
输出全1矩阵$C_1$如下（3行2列）：
$$C_1=\begin{bmatrix}
    [1., 1.]\\
    [1., 1.]\\
    [1. ,1.]\\
    \end{bmatrix}$$

### 2.3生成一个给定值的数组
函数:`tf.fill()`
参数:`dims:维度`、`value:给定的值`
用python生成一个给定值的矩阵$C_2$：
```python
#定义一个给定值的矩阵，形状为2行3列，值为6
C_2 = tf.Variable(tf.fill(dims=[2,3],value=6))
#用会话进行节点计算
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出生成的矩阵C_2
    print(sess.run(C_2))
```
输出给定值的矩阵$C_2$如下（2行3列）：
$$C_2=\begin{bmatrix}
    [6 ，6 ，6]\\
    [6 ，6 ，6]\\
    \end{bmatrix}$$
    
### 2.4生成一个给定值的常量
函数：`tf.constant()`,
参数：`value:给定值`，定值可以是一维向量，也可以是矩阵
用python生成一个给定值的常量$C_3$：
```python
#定义一个给定值的常量，值为[1,1,1,1]
C_3 = tf.Variable(tf.constant([1,1,1,1]))
#用会话进行节点计算
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出生成的矩阵C_3
    print(sess.run(C_3))
```
输出给定值的常量$C_3$如下：`[1 1 1 1]`

## End
说了那么多个生成函数，也差不多够用了，这些是比较常用的，熟记即可。

## 参考文献
- 《TensorFlow实战Google深度学习框架》
- 《人工智能实践：TensorFlow笔记》
































