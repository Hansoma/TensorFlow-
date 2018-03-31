#coding:utf-8


"""
tensorflow提供了七种偏置函数,
tf.nn.relu, tf.sigmoid, tf.tanh是常用的三种
同时tensorflow支持自定义偏置函数
下面代码给出几个使用样例
"""

"""
a = tf.nn.relu(tf.matmul(x, w1) + biases1)

y = tf.nn.relu(tf.matmul(a, w2) + biases2)
"""

