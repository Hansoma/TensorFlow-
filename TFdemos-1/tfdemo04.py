#coding:utf-8

"""
前向传播算法相关介绍
及神经网络参数与tensorflow变量
"""

import tensorflow as tf

"""
TensorFlow随机数生成函数
tf.random_normal() # 正态分布
tf.truncated_normal() # 正态分布,但是标准差超过2会被重新随机
tf.random_uniform() # 均匀分布
tf.constant() # 产生一个给定值的常量

TensorFlow常数生成函数
tf.zeros() # 生成全是0的数组
tf.ones() # 生成全是1的数组
tf.fill() # 产生一个全部为给定数字的数组
tf.constant() # 产生一个给定值的常量
"""

"""
# 声明一个2*3的矩阵变量
weights = tf.Variable(tf.random_normal([2,3], stddev=2))

# 声明一个偏直项,用常数来设置初始值
# 会生成一个初始值全部为0且长度为3的变量
biases = tf.Variable(tf.zeros([3]))

# w2的初始值被设置成和weights变量相同
w2 = tf.Variable(weights.initial_value())

# w3的初始值被设置成weights的初始值的两倍
w3 = tf.Variable(weights.initial_value() * 2.0)
"""

# 声明w1,w2两个变量,还通过seed参数设定了随机种子
# 这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal((2,3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3,1), stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量,注意这里的x是一个1x2的矩阵
x = tf.constant([[0.7, 0.9]])

# 通过前面所述的前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 与之前不同,不可以直接sess.run(y)来获取y的值
# 因为还要将w1, w2进行初始化

"""
w1, w2并没有被真正的运行,所以还需要通过initializer来进行初始化
特别的:对所有变量进行初始化
init_op = tf.global_variables_initializer()
sess.run(init_op)
"""

sess.run(w1.initializer) # 初始化w1
sess.run(w2.initializer) # 初始化w2

# 输出[]
print("*************Result of x**********************")
print(sess.run(x))
print("*************Result of w1***********************")
print(sess.run(w1))
print("*************Result of a*************************")
print(sess.run(a))
print("*************Result of w2***********************")
print(sess.run(w2))
print("*************Result of y************************")
print(sess.run(y))
"""
tf.glabal_variables()可以取出当前计算图中的所有变量
"""
print("*****************Result of all variables**********************")
print(sess.run(tf.global_variables()))

"""
通过声明函数中的trainable参数,来进行区分需要优化的参数
如果声明为True,则可以通过tf.trainable_variables函数得到所有的需要优化的参数
tensorflow会将GraphKeys.TRAINABLE_VARIABLES集合中的所有变量作默认的优化对象

"""


sess.close()













