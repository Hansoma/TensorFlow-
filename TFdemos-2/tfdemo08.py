#coding=utf-8
"""
1.tensorflow解决学习率的问题使用指数衰减法
    tf.train.exponential_decay
2.
"""

import tensorflow as tf

# 指数衰减法用法:
'''
tf.train.exponential_decay()实现了下面的算法
decayed_learning_rate = \
    learning_rate * decay_rate ^ (gobal_step / decay_steps)

decayed_learning_rate 为每一轮优化时使用的学习率,
learning_rate 为事先设计的初始学习率.
decay_rate 为衰减系数,
decay_steps 为衰减速度,

tf.train.exponential_decay()中可选参数:
staircase, 默认为False, 这时学习率的变化趋势如曲线,
如果设为True, global_step/decay_steps会被化为整数.
下面代码给出具体用法:
'''
global_step = tf.Variable(0)

# 通过exponential_dacay函数生成学习率
learning_rate = tf.train.exponential_decay(
    0.1, global_step, 100, 0.96, staircase=True)

# 使用指数衰减的学习率. 在minimize函数中传入global_step将自动更新global_step参数,
# 从而使得学习率也得到相应更新
"""
learning_step = tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(...my_loss..., global_step=global_step)
"""
"""
上面这段代码设定了初始学习率为0.1, 因为staircase=True所以每一百轮后学习率乘以0.96,
一般来说, 初始学习率, 衰减系数, 衰减速度都是根据经验设置的.
而且损失函数下降的速度和迭代结束后总损失的大小没有必然联系,
因此不能通过前几轮损失函数下降的速率来比较不同神经速率的效果.
"""

