#coding:utf-8
"""
tensorflow的数据模型:张量
张量的两种用途:
1.对中间计算结果的引用
2.当计算图构建完成之后,张量可以用来计算结果

tensorflow的运行模型:会话
通常通过python的上下文管理器来管理这个会话
"""
import tensorflow as tf
# tf.constant是一个计算,这个计算的结果为一个张量,保存在变量a中
a = tf.constant([1, 2], name="a", dtype=tf.float32) # 加数类型不同会报错,需指定dtype
b = tf.constant([2.0, 3.0], name="b")

result = a + b

with tf.Session() as sess:
    # 使用创建好的会话来计算结果
    print(sess.run(result))
    # 不需要调用Session.close函数,上下文退出时,会话也结束了
    # 同样功能:
    print(result.eval(session=sess))

# 通过tf.InteractiveSession方法可以省去将产生的会话注册为默认会话的过程
sess = tf.InteractiveSession()
print(result.eval())
sess.close()

# ConfigProto可以对session进行配置,类似线程数或是GPU分配策略
# 常用参数:allow_soft_placement: 一般设置为True, 在GPU无法进行运算时将运算转到CPU上.
#         log_device_placement: 记录log方便调试,生产过程中置为false减少日志量
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.InteractiveSession(config=config)

