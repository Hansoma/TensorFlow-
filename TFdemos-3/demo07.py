import tensorflow as tf
"""这个demo直接加载持久化的图"""
saver = tf.train.import_meta_graph(
    "/home/ma/PycharmProjects/TensorFlow-LearningLog/model/model.ckpt.meta"
)

with tf.Session() as sess:
    saver.restore(sess, "/home/ma/PycharmProjects/TensorFlow-LearningLog/model/model.ckpt")
    # 通过张量名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))