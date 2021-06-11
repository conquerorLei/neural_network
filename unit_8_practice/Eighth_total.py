import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from midterm import init as init

init.Initial.pltInitial()


def pltShow(acc_train, acc_test, cce_train, cce_test):
    # 结果可视化
    plt.figure(figsize=(10, 3))  # 创建画布

    plt.subplot(121)
    plt.plot(cce_train, color="blue", label="train")
    plt.plot(cce_test, color="red", label="test")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(122)
    plt.plot(acc_train, color="blue", label="train")
    plt.plot(acc_test, color="red", label="test")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def dataSolve():
    # 加载数据
    TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)  # 导入鸢尾花训练数据集
    TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)  # 导入鸢尾花测试数据集

    df_iris_train = pd.read_csv(train_path, header=0)  # 读取文件
    df_iris_test = pd.read_csv(test_path, header=0)
    # 数据预处理
    iris_train = np.array(df_iris_train)  # 转化为numpy数组
    iris_test = np.array(df_iris_test)

    x_train = iris_train[:, 0:4]  # 取训练集的全部属性
    y_train = iris_train[:, 4]  # 取最后一列标签值
    x_test = iris_test[:, 0:4]
    y_test = iris_test[:, 4]

    x_train = x_train - np.mean(x_train, axis=0)  # 对属性值进行标准化处理，使它均值为0
    x_test = x_test - np.mean(x_test, axis=0)

    X_train = tf.cast(x_train, tf.float32)  # 将属性值X转为32位浮点数
    Y_train = tf.one_hot(tf.constant(y_train, dtype=tf.int32), 3)  # 标签值Y转化为独热编码
    X_test = tf.cast(x_test, tf.float32)
    Y_test = tf.one_hot(tf.constant(y_test, dtype=tf.int32), 3)
    return X_train, X_test, Y_train, Y_test, y_train, y_test
