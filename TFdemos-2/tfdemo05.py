#coding=utf-8

"""
tf.where和tf.greater函数的用法
"""

import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()

#输出结果FFTT,前一个是否大于后一个
print(tf.greater(v1, v2).eval())

#类似C中的a>b?a:b
#输出结果:[4. 3. 3. 4.]
print(tf.where(tf.greater(v1, v2), v1, v2).eval())

sess.close()
