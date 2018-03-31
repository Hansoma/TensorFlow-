#coding=utf-8
"""
通过tensorflow训练神经网络模型

"""

import tensorflow as tf

"""
这个样例指定placeholder为一个3*2的矩阵,这样就可以得到3个样例的前向传播结果.
同时在feed_dic中做出改变,给出三个样例
"""


"""
通过placeholder实现前向传播算法
placeholder的类型也是不可以改变的
"""
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#定义placeholder作为存放数据的地方,这里的维度不一定要定义
#但如果维度是确定的,第=定义维度可以降低出错的概率

"""这里shape改为n*2"""
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#一定要给出feed_dict, 为palceholder设定取值
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

"""
得到batch的钱向传播数据之后,需要定义一个损失函数来刻画当前预测值与真实值之间的差距
以下代码定义了一个简单的损失函数,并通过tensorflow定义了前向传播算法
"""

"""
#使用sigmoid函数讲y转换为0~1之间的数值.转换后的y代表预测是正样本的概率,1-y代表预测是负样本的概率
y = tf.sigmoid(y)
#定义损失函数来刻画预测值与真实值的差距
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, le -10, 1.0))
    + (1-y)*tf.log(tf.clip_by_value(1-y, le-10, 1.0)))
#定义学习效率
learning_rate = 0.001
#定义反向传播算法来优化神经网络的参数
train_step = \
    tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
"""

"""
以上代码中,cross_entropy定义了真实值和预测值之间的交叉熵
第二行train_step定义了反向传播的优化方法.

"""