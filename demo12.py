import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "/home/ma/tfmodels/combined_model.pb"
    # 读取保存的模型文件, 并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前图中. return_elements=['add:0']给出了
    # 返回的张量得名称. 在保存的时候可以给出的是计算节点的名称, 所以为'add'. 在加载的时候,
    # 给出的是张量的名称, 所以是'add:0'
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])

    # 输出[3.0]
    print(sess.run(result))