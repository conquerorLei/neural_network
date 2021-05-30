import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from midterm import init as init
from matplotlib import colors as mc

init.Initial.pltInitial()
# 获取训练数据集
TRAIN_url = "http://download.tensorflow.org/data/iris_training.csv"

# 获取测试数据集
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
# 制定列标题
COLUMN_NAME = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
Index_property = [0, 2]
not_Index_species = 1


def getDataSet(train_url=TRAIN_url, test_url=TEST_URL, column_name=None):
    # TRAIN_URL.split('/')[-1]表示获取网址中的数据集文件名
    if column_name is None:
        column_name = COLUMN_NAME
    train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
    test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

    # Pandas访问csv数据集
    # pd读取csv数据
    df_iris_train = pd.read_csv(train_path, names=column_name, header=0)
    df_iris_test = pd.read_csv(test_path, names=column_name, header=0)
    # 转化成numpy数组
    iris_train = np.array(df_iris_train)
    iris_test = np.array(df_iris_test)
    return iris_train, iris_test


def getData(iris_train, iris_test, index_property=None, not_index_species=not_Index_species):
    # 属性
    if index_property is None:
        index_property = Index_property
    train_x = np.array([iris_train[:, i] for i in index_property]).T
    train_y = iris_train[:, 4]
    test_x = np.array([iris_test[:, i] for i in index_property]).T
    test_y = iris_test[:, 4]
    # 在属性的基础上获取种类
    x_train = train_x[train_y != not_index_species]
    y_train = train_y[train_y != not_index_species]
    x_test = test_x[test_y != not_index_species]
    y_test = test_y[test_y != not_index_species]
    return x_train, y_train, x_test, y_test


def displayPoint(x_train, y_train, x_test, y_test):
    plt.figure(10, 3)
    cm_pt = mc.ListedColormap(["blue", "red"])
    plt.subplot(121)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)
    plt.subplot(122)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_pt)
    plt.show()


def normalize(x_train, y_train, x_test, y_test):
    # displayPoint()
    # 数据归一化
    x_train = x_train - np.mean(x_train, axis=0)
    x_test = x_test - np.mean(x_test, axis=0)
    # displayPoint()
    x0_train = np.ones(len(x_train)).reshape((-1, 1))
    x0_test = np.ones(len(x_test)).reshape((-1, 1))
    mx_train = tf.cast(tf.concat((x0_train, x_train), axis=1), dtype=tf.float32)
    my_train = tf.cast(y_train.reshape((-1, 1)), dtype=tf.float32)
    mx_test = tf.cast(tf.concat((x0_test, x_test), axis=1), dtype=tf.float32)
    my_test = tf.cast(y_test.reshape((-1, 1)), dtype=tf.float32)
    return mx_train, my_train, mx_test, my_test


def trainModel(mx_train, my_train, mx_test, my_test, learn_rate=0.01, my_iter=120, display_step=30, rand=3):
    # 多元逻辑回归

    np.random.seed(612)
    w = tf.Variable(np.random.randn(rand, 1), dtype=tf.float32)

    ce_train = []
    ce_test = []
    acc_train = []
    acc_test = []

    for i in range(0, my_iter + 1):
        with tf.GradientTape() as tape:
            pred_train = 1 / (1 + tf.exp(-tf.matmul(mx_train, w)))
            loss_train = -tf.reduce_mean(my_train * tf.math.log(pred_train) + (1 - my_train) * tf.math.log(1 - pred_train))
            pred_test = 1 / (1 + tf.exp(-tf.matmul(mx_test, w)))
            loss_test = -tf.reduce_mean(my_test * tf.math.log(pred_test) + (1 - my_test) * tf.math.log(1 - pred_test))
        accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train.numpy() < 0.5, 0., 1.), my_train), tf.float32))
        accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_test.numpy() < 0.5, 0., 1.), my_test), tf.float32))

        ce_train.append(loss_train)
        ce_test.append(loss_test)
        acc_train.append(accuracy_train)
        acc_test.append(accuracy_test)

        dl_dw = tape.gradient(loss_train, w)
        w.assign_sub(learn_rate * dl_dw)

        if i % display_step == 0:
            print("i: %i, TrainAcc: %f, TestAcc: %f, TrainLoss: %f, TestLoss: %f" % (i, accuracy_train, accuracy_test, loss_train, loss_test))
    return ce_train, ce_test, acc_train, acc_test, w


def show(ce_train, ce_test, acc_train, acc_test):
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    plt.plot(ce_train, color="blue", label="train")
    plt.plot(ce_test, color="red", label="test")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(acc_train, color="blue", label="train")
    plt.plot(acc_test, color="red", label="test")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
