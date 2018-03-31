# coding=utf-8
"""
TensorFlow中有一个类专门用于处理MNIST数据,
下面是演示demo
"""

from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集, 如果指定的地址下没有测试数据, 那么tensorflow会自动下载
mnist = input_data.read_data_sets("/home/ma/PycharmProjects/MNIST_data", one_hot=True)

# 打印Training data size: 55000
print("Training data size: ", mnist.train.num_examples)

# 打印Validating data size: 5000
print("Validating data size: ", mnist.validation.num_examples)

# 打印Testing training size: 10000
print("Testing data size: ", mnist.test.num_examples)

# 打印Example training data: [...]
print("Example training data: ", mnist.train.images[0])

# 打印Example training data label:
print("Example training data label: ", mnist.train.labels)

# 为了方便使用随机梯度下降, input_data.read_data_sets函数生成的类还提供了mnist.train.next_batch函数
# 它可以从所有的训练数据中读取一小部分作为一个训练batch
# 以下代码展示了如何使用这个功能:
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train集合中选取batch_size个训练数据
print("X shape :", xs.shape)
# 输出X shape: (100, 784)
print("Y shape :", ys.shape)
# 输出Y shape: (100, 10)

