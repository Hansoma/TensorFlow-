#coding=utf-8

"""
这个demo介绍过拟合问题

详情见书本page87 88

为了解决过拟合问题, 不直接优化J(Θ), 而是优化J(Θ)+λR(w),

其中, R(w)刻画的是模型的复杂程度, 而λ表示模型复杂损失在总损失中的比例.

注意这里的Θ表示的是一个模型中的所有参数, 包括边的权重w和偏置项b

常用刻画模型复杂度的函数有两种,详情见书本page88

"""

import tensorflow as tf

"""
下面的代码给出了一个简单的带L2的正则化的损失函数定义
"""
"""
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)


loss = tf.reduce_mean(tf.square(y_ - y)) +
        tf.contrib.layers.l2regularrizer(lambda)(w)

"""

"""
下面的代码给出了L1和L2正则化的使用样例
"""
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    """0.5为正则化的权重"""
    # (|1| + |-2| + |-3| + |4|)*0.5 = 5.0
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    # (1^2 + (-2)^2 + (-3) ^2 + 4^2)/2 * 0.5 = 7.5
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))

"""
在简单的神经网络中,这样的方式就可以很好的计算带有正则化的损失函数了.
但当神经网络的参数增多之后, 这样的方式:
1.可能导致loss的定义很长.
2.网络结构复杂之后, 定义网络结构的部分和计算损失函数的部分可能不在同一个函数中,
    这样通过变量这种方式计算损失函数就不方便了.
"""