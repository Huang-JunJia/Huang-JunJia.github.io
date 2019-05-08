---
layout:     post
title:      "TensorFlow之初见端倪"
subtitle:   "TensorFlow是由Google开发的一个用作深度学习的计算框架，这个开源的计算框架可以很好地实现各种深度学习算法。"
date:       2019-04-21
author:     "木夏"
header-img: "img/post-bg-digital-native.jpg"
catalog: true
mathjax: true
tags:
    - TensorFlow

---

>`那时我们有梦， 关于文学、关于爱情、关于穿越世界的旅行。`<br/>`如今我们深夜饮酒，杯子碰到一起，都是梦破碎的声音。——北岛《波兰来客》`

## Begin
## 1.TensorFlow简介
TensorFlow是由Google开发的一个用作深度学习的计算框架，这个开源的计算框架可以很好地实现各种深度学习算法。当然，用作深度学习的框架不止只有TensorFlow，还有蒙特利尔大学开发的**Theano**框架、加州大学伯克利分校开发的**Caffe**框架，这些都是比较热门的深度学习框架。

## 2.TensorFlow基础
TensorFlow是用张量（tensor）表示数据，用计算图表示搭建神经网络，用会话（session）执行计算图，优化线上的权重（weight），得到模型。以下来介绍一些基础概念及其用法。
### 2.1计算图(graph)
计算图是TensorFlow的一个最基本的概念，TensorFlow中所有的计算都会被转化成计算图上的节点。
#### 2.1.1计算图的使用
在TensorFlow中，系统会自动维护一个默认的计算图，可以通过`tf.get_default_graph`获取当前默认的计算图。
```python
#当前默认的计算图,以下输出为True
print( a.graph is tf.get_default_graph)
```
还可以使用`tf.Graph`函数来生成新的计算图，注意，不同计算图上的张量和运算不会共享。
```python
g1 = tf.Graph()
with g1.as_default():
    #定义变量v，并且赋值为0
    v = tf.get_variable("v", initializer=tf.zeros_initializer()(shape =[1]))
#放到会话（session)中，才能被运行生效。会话后面会讲。
with tf.Session(graph= g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print( sess.run(tf.get_variable("v")))
#输出v为[0.]
```
上面代码的计算图g1中，变量`v`被赋值为0，所以输出变量`v`为[0.]
```
g2 = tf.Graph()
with g2.as_default():
    #定义变量v，并且赋值为1
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape =[1]))
#放到会话（session)中，才能被运行生效
with tf.Session(graph= g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print( sess.run(tf.get_variable("v")))
#输出v为[1.]
```
上面代码的计算图g2中，相同的变量`v`被赋值为1，所以输出变量`v`为[1.].对比可见，不同的计算图上的张量和运算并不会被共享。
### 2.2张量(tensor)
在TensorFlow中，所有的数据都用张量（tensor）表示。张量可以表示0阶~N阶的数组（列表），其中的`阶`即为张量的维度。
#### 2.2.1张量维度介绍
| 维度        | 阶   |  名字  |列子|
| --------   | -----:  | :----:  | :----:  |
| 0维     | 0 |   标量(scalar)     |v = 100, v=abc
| 1维   |   1   |   向量(vector)   |v=[1,2,3],一维数组
| 2维       |    2   |  矩阵(matrix) |v=[[1,2,3], [4,5,6],[7,8,9]]
| N维   |   N  |  张量(tensor)  |v=[[[[[....(N个[ 方括号，即N维)
#### 2.2.2张量属性
一个张量主要保存三个属性：`名字（name）`、`维度(shape）`、`类型(type)`
```
#定义张量a,b
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([3.0,4.0],name='b')
result = tf.add(a,b,name="add")
#也可以这样写:result = a+b
print(result)
```
输出结果如下:
```
Tensor("add:0", shape=(2,), dtype=float32)
```
其中`add:0`为结点名，表示第0个输出；`shape=(2,)`表示一个一维长度为2的数组；`dtype=float32`表示数据类型为32位浮点数。
**注意**：如果想看最后的`result`的结果，需要运用会话（session）,才会显示最终结果，这现在只是中间存储的张量，张量本身不存储具体的数字。
#### 2.2.3利用张量构建一个神经元的计算过程
神经元如下图所示，$y=XW=x_1*w_1+x_2*w_2$
![此处输入图片的描述][1]
将$x_1、x_2$的值放入一个1x2（1行2列）的矩阵$X$中，将$w_1、w_2$的值放入2x1（2行1列）的矩阵$W$，将矩阵$X$点乘矩阵$W$，最后结果就是一个数，即$y$值。
代码如下：
```
#将x1、x2的值放入一个1x2（1行2列）的矩阵X中
x = tf.constant([[1.0,2.0]],name='x')
#将w1、w2的值放入2x1（2行1列）的矩阵W
w = tf.constant([[3.0],[4.0]],name='w')
#两个矩阵相乘  
y = tf.matmul(x,w,name='y')
print(y)
```
输出的结果（只是输出张量，并非最后的神经元的结果，最后的神经元结果请看2.3.1）:
```
Tensor("y:0", shape=(1, 1), dtype=float32)
```
### 2.3会话（session)
会话（session）拥有并管理TensorFlow程序运行时的所有资源，当所有计算完后之后，需要关闭会话，否则可能会出现内存泄漏。
#### 2.3.1建立会话，进行节点运算
利用with语句，即可建立会话，这样不必手动关闭会话，系统在完成计算时会自动关闭会话。下面是对$y$进行运算：
```
with tf.Session() as sess:
    sess.run(y)         #运行y
    print(sess.run(y))  #输出运行的y  
```
输出的结果:
`[[11.]]`
这个值，就是$y$值，即2.2.3中神经元的值，是矩阵$X$和矩阵$W$相乘的值。
#### 2.3.2利用会话对变量初始化
所有变量的初始化都需要在会话中先进行，然后才能进行张量运算。
```
init_op = tf.global_variables_initializer()
sess.run(init_op) 
```
### 2.4占位符（placeholder）
#### 2.4.1为什么需要占位符？
因为底层语言执行一行脚本，就要切换一次，这样是有成本开销的。所有我们在构建计算图的时候，用`tf.placeholder()`进行占位，此时并没有把数据喂入模型，只是把分配相应的内存资源，以便后面可以把数据喂入模型。
#### 2.4.2占位符的使用
首先使用`tf.placeholder()`进行分配内存，进行占位；最后利用`feed_dict = {}`，将数据喂入模型。代码如下图：
```
#占位符
input1 = tf.placeholder(tf.float32)#占位符input1，未赋值，类型：tf.float32
input2 = tf.placeholder(tf.float32)#占位符input2，未赋值，类型：tf.float32

output = tf.multiply(input1, input2) #输出output为input1和input2相乘

with tf.Session() as sess:#建立会话
    #利用feed_dict{}将数据input1，input2喂入模型，载运行
    print(  sess.run(output, feed_dict={input1:[5.],input2:[6.]}) )
```
最后输出：
`[30.]`

## End
了解`计算图（graph）`、`张量（tensor）`、`会话（session）`、`占位符（placeholder）`这几个基本概念，其中我们也知道了一个神经元如何来表示，那么神经网络就是由很多个神经元组成的[（2.2.3）](#223-利用张量构建一个神经元的计算过程)，这样来说，我们就可以建立一个简单的神经网络了。


## 参考文献：
- 《TensorFlow实战Google深度学习框架》
- 《人工智能实践：TensorFlow笔记》



  [1]: https://s2.ax1x.com/2019/05/07/EyYtwn.png
