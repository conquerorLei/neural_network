import tensorflow as tf
import numpy as np
from unit_8_practice import Eighth_total as et

X_train, X_test, Y_train, Y_test, y_train, y_test = et.dataSolve()

# 设置超参数，迭代次数和显示间隔
learn_rate = 2.33
iter = 100
display_step = 1
# 设置模型参数初始值
np.random.seed(612)
W = tf.Variable(np.random.randn(4, 3), dtype=tf.float32)
B = tf.Variable(np.zeros([3]), dtype=tf.float32)
# 训练模型
acc_train = []
acc_test = []
cce_train = []
cce_test = []
for i in range(0, iter + 1):
    with tf.GradientTape() as tape:
        PRED_train = tf.nn.softmax(tf.matmul(X_train, W) + B)
        Loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train, y_pred=PRED_train))
    PRED_test = tf.nn.softmax(tf.matmul(X_test, W) + B)
    Loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test, y_pred=PRED_test))
    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(), axis=1), y_train), tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(), axis=1), y_test), tf.float32))
    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
    cce_train.append(Loss_train)
    cce_test.append(Loss_test)
    grads = tape.gradient(Loss_train, [W, B])
    W.assign_sub(learn_rate * grads[0])
    B.assign_sub(learn_rate * grads[1])
    if i % display_step == 0:
        print("i:%i,TrainAcc:%f,TrainLoss:%f,TestAcc:%f,TestLoss:%f" % (
            i, accuracy_train, Loss_train, accuracy_test, Loss_test))

et.pltShow(acc_train, acc_test, cce_train, cce_test)
