#coding=utf-8
"""
这个demo介绍如何通过反向传播算法和梯度下降算法
调整神经网络中参数的取值
反向传播算法是训练神经网络的核心算法,
它可以根据定义好的损失函数优化神经网络中参数的取值,
从而使神经网络模型在训练数据集上的损失函数达到最小值

"""
import tensorflow as tf

"""
在实际应用中,常采用综合梯度下降算法和随机梯度下降算法的折中,
即每次计算一小部分数据.
这一小部分数据被成为batch
本书的所有样例中,神经网络的训练都大致遵循下面的流程.
demo如下:
"""

batch_size = n

# 每次读取一小部分数据作为当前的训练数据来执行反向传播算法
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

# 定义神经网络结构和优化算法
loss = ...
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 训练神经网络
with tf.Session() as sess:
    # 参数初始化
    ...
    # 迭代的更新参数
    for i in range(STEPS):
        # 准备batch_size个训练数据. 一般将所有的训练数据随机打乱之后再选取可以
        # 得到最好的优化效果
        current_X, current_Y = ...
        sess.run(train_step, feed_dict={x: current_X, y: current_Y})

