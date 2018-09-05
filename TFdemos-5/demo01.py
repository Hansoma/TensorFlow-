"""这个demo用于演示tensorflow实现卷积层的步骤"""
import tensorflow as tf
# 通过tf.get_variable 的方式创建过滤器的权重变量和偏置项变量, 
# 上面介绍了卷积层的参数个数只和过滤器的尺寸, 深度, 以及当前层节
# 点矩阵的深度有关, 所以这里声明的参数变量是一个四维矩阵, 前两个
# 维度代表了过滤器的尺寸, 第三个维度代表当前层的深度, 第四个维度
# 表示过滤器的深度. 
filter_weight = tf.get_variable(
    'weights', [5, 5, 3, 16], 
    initializer=tf.truncated_normal_initializer(stddev=0.1)
)

# 和卷积层的权重类似, 当前层矩阵上不同位置的偏置项也是共享的, 所以总共有
# 下一层深度个 不同的偏置项. 本样例代码中16为过滤器深度, 也是神经网络中
# 下一层节点矩阵的深度.

biases = tf.get_variable(
    'biases', [16], initializer=tf.constant_initializer(0.1)
)

# tf.nn.conv2d() 提供了一个非常方便的函数来实现卷积层前向传播的方法, 
# 这个函数的第一个输入为当前层的节点矩阵, 注意这个矩阵是一个四维矩阵, 
# 后面三个维度对应一个节点矩阵, 第一维 对应 一个输入的batch, 比如在
# 输入层, input[0, :, :, :]表示第一张图片
# 第二个参数提供卷积层的权重,
# 第三个参数为不同维度上的步长, 
# 最后一个参数是填充(padding)的方法, 有SAME和VALID两种选择. 

conv = tf.nn.conv2d(
    input, filter_weight, strides=[1, 1, 1, 1], padding='SAME'
)

# tf.nn.bias_add 提供了一个方便的函数给每一个节点加上偏置项. 注意这里不能
# 直接使用加法, 因为矩阵上不同位置上的节点都需要加上同样的偏置项. 

bias = tf.nn.bias_add(conv, biases)

# 将计算结果通过RELU 激活函数完成去线性化
actived_conv = tf.nn.relu(bias)

# tf.nn.max_pool 实现了最大池化层的前向传播过程, 它的参数和tf.nn.conv2d
# 函数类似. ksize提供了过滤器的尺寸, strides提供了步长信息, padding提供了
# 是否使用全0填充.
pool = tf.nn.max_pool(actived_conv, ksize = [1, 3, 3, 1], 
                      strides=[1, 2, 2, 1], padding='SAME'  
)

