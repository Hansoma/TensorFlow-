#coding=utf-8
"""
通过tensorflow训练神经网络模型

"""

import tensorflow as tf

"""
通过placeholder实现前向传播算法
placeholder的类型也是不可以改变的
"""
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#定义placeholder作为存放数据的地方,这里的维度不一定要定义
#但如果维度是确定的,第=定义维度可以降低出错的概率

x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#一定要给出feed_dict, 为palceholder设定取值
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
