import tensorflow as tf

v1 = tf.get_variable("v1", [1])
print(v1.name) # 输出v1:0 表示了这个变量是生成变量这个运算的地一个结果

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name) # 输出foo/v:0, tf通过/来分隔命名空间的名称和变量的名称

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name) # 输出foo/bar/v:0

    v4 = tf.get_variable("v1", [1])
    print(v4.name) # 输出foo/v1:0 命名空间退出之后就不会再加前缀了.

# 创建一个名称为空的命名空间, 并设置reuse = True
with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1]) # 可以直接通过带命名空间的变量名
                                            # 来获取其他明明空间下的变量.
    print(v5 == v3)

    v6 = tf.get_variable("foo/v1", [1])
    print(v6 == v4)

