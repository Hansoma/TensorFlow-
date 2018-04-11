import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用这个函数可以直接生成上面代码中提供的字典.
print(ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    saver.restore(sess, "/home/ma/tfmodels/model.ckpt")
    print(sess.run(v))