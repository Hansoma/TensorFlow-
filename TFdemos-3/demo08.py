import tensorflow as tf

"""这个demo展示如何保存滑动平均模型"""

v = tf.Variable(0, dtype=tf.float32, name="v")

# 在没有声明滑动平均模型时, 只有一个变量v, 所以下面输出"v:0"
for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在声明滑动平均模型之后, TensorFlow会自动生成一个影子变量.
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时, TensorFlow会将v:0和v/ExponentialMovingAverage:0 两个变量都保存下来.
    saver.save(sess, "/home/ma/tfmodels/model.ckpt")
    print(sess.run([v, ema.average(v)]))

