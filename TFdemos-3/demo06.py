import tensorflow as tf

"""这个demo演示如何加载已经保存过的模型."""

# 使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载已经保存过的模型.
    saver.restore(sess, "/home/ma/PycharmProjects/TensorFlow-LearningLog/model/model.ckpt")
    print(sess.run(result))

