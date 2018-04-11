import tensorflow as tf
"""以下代码展示了如何通过变量重命名直接读取滑动平均的值.
下面这段代码中, 读取的变量v实际上是上面代码中v的滑动平均值,]
通过这个方法可以使用完全一样的代码来计算滑动平均模型的钱想传播计算结果.
"""
v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名来将原来的变量v的滑动平均值直接赋值给v.
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "/home/ma/tfmodels/model.ckpt")
    print(sess.run(v))