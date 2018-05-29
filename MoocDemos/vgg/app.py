#coding:utf-8
import numpy as np

# Linux服务器没有GUI的情况下使用matplotlib绘图, 必须至于pyplot之前
# import matplotlib
# matplotlib.use("Agg")

import tensorflow as tf
import matplotlib.pyplot as plt

# 引用三个自定义模块
import vgg16
import utils
from Nclasses import labels

img_path = input('Input the path and image name:')
img_ready = utils.load_image(img_path) # 调用load_image函数, 对待测试图片做一些预处理操作

# 定义一个fugure画图窗口, 并指定窗口名称, 也可以设置窗口的大小
fig=plt.figure(u"Top-5 预测结果") 

with tf.Session() as sess:
    # 定义一个维度为[1, 224, 224, 3], 类型为float32的tensor占位符
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    # 调用类的成员方法forward(), 并传入待测试图像, 这也就是前向传播的过程
    vgg.forward(images)
    # 将一个batch的数据喂入网络, 得到网络的预测输出
    probability = sess.run(vgg.prob, feed_dict={images:img_ready})
    # np.argsort函数返回预测值(probability的数据结构:[[各预测类别的概率值]]) 由小到大的索引
    # 并取出预测概率最大的五个索引值
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print ("top5:",top5)
    # 定义两个list对应的概率值和实际标签
    values = []
    bar_label = []

    for n, i in enumerate(top5): # 枚举上面取出的五个索引值
        print ("n:",n)
        print ("i:",i)
        values.append(probability[0][i]) # 将索引值对应的预测概率值取出并放入 values
        bar_label.append(labels[i]) # 根据索引值取出对应的实际标签并放入 bar_label
        print (i, ":", labels[i], "----", utils.percent(probability[0][i])) # 打印出属于某个类别的概率
        
    ax = fig.add_subplot(111) # 将画布划分为一行一列, 并把下图放入其中
    # bar() 函数绘制柱状图, 参数range(len(values))是柱子下标, values表示柱高的列表(也就是五个预测概率值,
    # tick_label 是每个柱子上显示的标签(实际对应的标签), width是柱子的宽度, fc是柱子的颜色)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
    ax.set_ylabel(u'probabilityit') # 设置横轴标签
    ax.set_title(u'Top-5') # 添加标签
    for a,b in zip(range(len(values)), values):
        # 在每个柱子的顶端添加对应的预测值概率, a, b表示做高, b+0.0005表示要把文本信息放置在高于每个柱子顶端0.0005的位置
        # center 表示每个文本位于柱子顶端水平方向上中间位置, bottom是将文本水平放置在柱子顶端垂直方向上的低端位置, fontsize是字号
        ax.text(a, b+0.0005, utils.percent(b), ha='center', va = 'bottom', fontsize=7)

    plt.savefig("result.jpg") # 保存图片

    plt.show() # 弹窗展示图像(在服务器上将这句注释)


    
