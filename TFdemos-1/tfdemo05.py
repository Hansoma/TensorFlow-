#coding:utf-8

"""
神经网络参数和tensorflow变量
"""

import tensorflow as tf

"""
变量与张量的关系:
tfdemo4中,tf.Variable是一个运算,
这个运算的输出结果是一个张量.
"""

"""
张量,维度和类型是变量的最重要属性,
变量的类型是不可变的,一个变量在构建之后,它的类型就不能再改变了,
维度在程序中是可以改变的.
需要通过参数设置validate_shape=False,例如下面这段代码:
"""

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name="w1")
w2 = tf.Variable(tf.random_normal([2, 2], stddev=1), name="w2")

#下面的这段会报错:维度不匹配
#tf.assign(w1, w2)

#改成这一句可以成功执行:
#注意: 这种用法在实践中比较少见
tf.assign(w1, w2, validate_shape=False)