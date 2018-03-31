#coding:utf-8

"""
tensorflow计算模型--计算图
计算图的简单使用
"""

import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量v,并设置初始值为0
    v = tf.get_variable(
        "v", initializer=tf.zeros_initializer()(shape=[1])
    )

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量v,并设置初始值为1
    v = tf.get_variable(
        "v", initializer=tf.ones_initializer()(shape=[1])
    )

# 在计算图g1中读取变量v1的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # 计算图v1中,变量v的值为0,所以下面这行会输出[0.]
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量v2的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g2中,变量v的值应当为1,所以下面输出[1.]
        print(sess.run(tf.get_variable("v")))

"""计算图可以通过tf.Graph.device来指定运行计算的设备
    例如:
    见tfdemo02.py
"""