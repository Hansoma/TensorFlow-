import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""以下代码给出神经网络的训练程序"""

# 加载mnist_inference.py 中定义的常量和前向传播函数
import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径
MODEL_SAVE_PATH = "/home/ma/PycharmProjects/MNIST_models/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输出的placeholder.
    x = tf.placeholder(
        tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input'
    )
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用之前定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 和5.2.1节样例中类似的定义损失函数, 学习率, 滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variable_averages_op =  variable_averages.apply(
        tf.trainable_variables()
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
                .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensor持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现, 验证和
        # 测试的过程将会有一个独立的程序来完成.
        for  i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})

            # 每一千轮保存一个模型
            if i % 1000 == 0:
                # 输入当前的训练情况, 这里只输入了模型在当前训练batch上的损失函数大小.
                # 通过损失函数的大小可以大概了解训练情况, 在验证数据集上的正确率会有一个单独的程序来生成
                print("After {} training steps, loss on training "
                      "batch is {}".format(step, loss_value))
                # 保存当前的模型, 注意这里给出了global_step参数, 这样可以让
                # 每个被保存的模型的文件名末尾加上训练轮数,
                # 比如: "model.ckpt-1000"
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
def main(argv=None):
    mnist = input_data.read_data_sets("/home/ma/PycharmProjects/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()