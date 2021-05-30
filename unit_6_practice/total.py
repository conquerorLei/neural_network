import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import time

from midterm import init as init

init.Initial.pltInitial()


def normalize(train_x, train_y, test_x, test_y):
    # 归一化处理
    x_train = (train_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
    y_train = train_y

    x_test = (test_x - test_x.min(axis=0)) / (test_x.max(axis=0) - test_x.min(axis=0))
    y_test = test_y

    # 堆叠
    x0_train = np.ones(len(train_x)).reshape((-1, 1))
    x0_test = np.ones(len(test_x)).reshape((-1, 1))

    # 转化为张量
    X_train = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)
    X_test = tf.cast(tf.concat([x0_test, x_test], axis=1), tf.float32)

    Y_train = tf.constant(y_train.reshape((-1, 1)), tf.float32)
    Y_test = tf.constant(y_test.reshape((-1, 1)), tf.float32)

    return X_train, Y_train, X_test, Y_test


def trainModel(X_train, Y_train, X_test, Y_test, rand=14, rate=0.01, miter=2000, ds=200):
    # 设置超参数
    learn_rate = rate
    my_iter = miter
    display_step = ds

    # 设置模型变量初始值
    np.random.seed(612)
    W = tf.Variable(np.random.randn(rand, 1), dtype=tf.float32)

    # 训练模型
    mse_train = []
    mse_test = []
    start = time.time()
    for i in range(0, my_iter + 1):
        with tf.GradientTape() as tape:
            Pred_train = tf.matmul(X_train, W)
            Loss_train = 0.5 * tf.reduce_mean(tf.square(Y_train - Pred_train))

            Pred_test = tf.matmul(X_test, W)
            Loss_test = 0.5 * tf.reduce_mean(tf.square(Y_test - Pred_test))

        mse_train.append(Loss_train)
        mse_test.append(Loss_test)

        dL_dW = tape.gradient(Loss_train, W)
        W.assign_sub(learn_rate * dL_dW)

        if i % display_step == 0:
            print("i: %i, Train Loss: %f, Test Loss: %f" % (i, Loss_train, Loss_test))
    end = time.time()
    print("time cast: %fs" % (end-start))

    return mse_train, mse_test, Pred_train, Pred_test, W


def show(mse_train, mse_test, y_train, y_test, Pred_train, Pred_test):
    # 可视化输出
    plt.figure(figsize=(20, 4))

    plt.subplot(131)
    plt.ylabel("MSE")
    plt.plot(mse_train, color="blue", linewidth=3)
    plt.plot(mse_test, color="red", linewidth=1.5)

    plt.subplot(132)
    plt.ylabel("Price")
    plt.plot(y_train, color="blue", marker="o", label="true_price")
    plt.plot(Pred_train, color="red", marker=".", label="pred_price")
    plt.legend()

    plt.subplot(133)
    plt.ylabel("Price")
    plt.plot(y_test, color="blue", marker="o", label="true_price")
    plt.plot(Pred_test, color="red", marker=".", label="pred_price")
    plt.legend()

    plt.show()


def outExcel(train, test, path="", filename=""):
    if len(path) == 0:
        ew = pd.ExcelWriter("../resource/"+filename+"Out.xlsx")
    else:
        ew = pd.ExcelWriter(path)
    df = pd.DataFrame({"train_loss": train, "test_loss": test})
    df.to_excel(ew, sheet_name="sheet1")
    ew.save()

