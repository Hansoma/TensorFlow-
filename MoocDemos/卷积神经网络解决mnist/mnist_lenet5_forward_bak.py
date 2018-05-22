import tensorflow as tf
IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):
    """卷积函数"""
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    """池化函数
    池化核大小是2x2, 步长也为2
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def forward(x, train, regularizer):
    # 初始化第一层卷积核
    conv1_w = get_weight(
        shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM],
        regularizer=regularizer)
    # 初始化第一层偏置
    conv1_b = get_bias(shape=[CONV1_KERNEL_NUM])
    # 卷积计算, 卷积核是初始化的conv1_w
    conv1 = conv2d(x, conv1_w)
    # 对 conv1 偏置激活
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 再池化
    pool1 = max_pool_2x2(relu1)

    # 第二层卷积核的深度等于上一层卷积核的个数, 即 CONV1_KERNEL_NUM
    conv2_w = get_weight(
        shape=[CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],
        regularizer=regularizer)
    # 初始化第二层的偏置
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # 卷积计算, 第二层的输入是第一层的输出, 即pool1
    conv2 = conv2d(pool1, conv2_w)
    # 激活
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    # 池化, pool2 为第二层的输出, 需要将它由三维张量变为二维张量
    pool2 = max_pool_2x2(relu2)

    # 将pool2的维度存入pool_shape中
    pool_shape = pool2.get_shape().as_list()
    # 提取特征的长度, 宽度, 深度, 相乘得到所有特征点的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # pool_shape[0]是一个batch的值
    # 将pool2 提取为batch行, nodes列
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # pool2喂入全连接网络
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    # 第二层全连接神经网络.
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y