import tensorflow as tf
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
g = tf.Graph()
#指定计算运行的设备
with g.device('/gpu:0'):
    result = a + b
