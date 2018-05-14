import numpy as np 
import matplotlib.pyplot as plt 

seed = 2
def generate():
    # 基于seed产成随机数
    rdm = np.random.RandomState(seed)
    # 随机数返回300行2列的矩阵, 表示300组坐标点.作为输入数据输入数据集.
    X = rdm.randn(300, 2)
    # 从X这个矩阵中取出一行, 如果两个坐标平方和小于2, 给Y赋值1, 其余赋值0
    # 作为输入数据集的标签
    Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]
    # 遍历Y中每个元素, 1 赋值red, 其余赋值blue
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对数据集X和标签Y进行形状整理, 第一个元素用-1表示跟随第二列计算, 第二列元素表示多少列,
    # X为2列, Y为1列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    return X, Y_, Y_c

