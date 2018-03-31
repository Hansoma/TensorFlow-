#coding = utf-8

"""
下面这段代码,通过一个简单的神经网络来讲解损失函数对训练结果的影响.
下面这段实现了一个拥有两个输入节点,一个输出节点没有隐藏层的神经网络.
这个程序的主题流程和TFdemos-1中的demo07类似,但用到了前一个demo中定义的损失函数.
"""

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义了一个单层的神经网络前向传播的过程, 这里就是简单的加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

"""自定义损失函数如下:"""
# 定义预测多了和少了的成本
loss_less = 10
loss_more = 1
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), # y是否大于y_, 预测自是否大于实际值
                               (y - y_) * loss_more, # 预测值大于实际值的情况,loss_more
                               (y_ - y) * loss_less, # 预测值小于实际值的情况,loss_less
                               ))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置回归的正确值为两个输入加上一个随机量. 之所以要加上一个随机量是为了加入不可预测的噪音,
# 否则不同损失函数的意义就不大了, 因为不同的损失函数都会在能完全预测正确的时候最低.
# 一般来说,噪音为一个均值为0的小量, 所以这里的噪音设置为-0.05~0.05的随机数
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for x1, x2 in X]

#训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        # 我的输出结果是[[1.0193471] [1.0428091]],这比x1 + x2要大,也就是说,
        # 模型倾向于预测多一点.
        # 如果将loss_less 和 loss_more的值互换, 模型会倾向于预测少一点.
        print(sess.run(w1))







