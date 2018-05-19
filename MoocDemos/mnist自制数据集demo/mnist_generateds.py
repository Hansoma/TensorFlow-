import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = 'path_to_jpg'
label_train_path = 'path_to_label'
tfRecord_train = ''
image_test_path = 'path_to_test_image'
label_test_path = 'label_to_test'
tfRecord_test = ''
data_path = 'path_to_data'
resize_height = 28
resize_width = 28

def write_tfRecord(tfRecordName, image_path, label_path):
    """按照tf.train.Example写入数据集"""
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    # 读取label
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    # 遍历文件中的图片名称, label, 为制作数据集做准备
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1
        # 按照模板制作example
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializerToString())
        num_pic += 1
        print('The number of picture is ', num_pic)
    writer.close()
    print("Write tfrecord successful")

def read_tfRecord(tfRecord_path):
    """读取tfRecord文件"""
    # 文件名队列, 包括所有的数据集, 这里只有一个数据集.
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    # 新建reader, 将读取的内容保存到serialized_example中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 解序列化, 键名与制作时的键明相同
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64), # 标签是10分类
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    # 将img_raw解为8位无符号整型
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 变为1行784列
    img.set_shape([784])
    # 将元素变为浮点数
    img = tf.cast(img, tf.float32) * (1 / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label

def get_tfRecord(num ,isTrain=True):
    """批读取tfRecords的代码
    isTrain: 训练集为True, 测试集为False
    """
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch( # 该函数会从总样本中取出capacity组数据并打乱
                        [img, label],
                        batch_size = num,
                        num_threads = 2, # 整个过程使用了2个线程
                        capacity = 1000,
                        min_after_dequeue = 700)
    return img_batch, label_batch



def generate_tfRecord():
    """制作数据集"""
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The directory was created successfully")
    else:
        print("Directory already exists.")

    write_tfRecord(tfRecord_train, image_train_path)
    write_tfRecord(tfRecord_test, image_test_path)

def main():
    generate_tfRecord()

if __name__ == "__main__":
    main()