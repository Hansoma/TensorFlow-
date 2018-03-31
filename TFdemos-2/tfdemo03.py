#coding=utf-8
"""
损失函数的定义

"""

import tensorflow as tf

"""
经典损失函数
本节介绍分类问题和回归问题中使用的经典损失函数
"""
"""
下面这段代码展示了如何计算交叉熵
"""
sess = tf.Session()

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.reduce_mean(v).eval(session=sess))

"""
因为交叉熵一般会和softmax一起使用,所以Tensorflow对这两个功能进行了合并
下面进行实现:
这样通过一个命令就可以得到使用softmax回归之后的交叉熵
"""

#y代表原始神经网络的输出结果
#y_代表标准答案
y = []
y_ = []
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=y_, logits=y)

"""
回归问题解决的是对具体数字的预测,
常用的损失函数是均方误差(MSE, mean squared error)
以下代码展示了如何实现均方误差损失函数
"""
y = 0
y_ = 0
mse = tf.reduce_mean(tf.square(y_ - y))

