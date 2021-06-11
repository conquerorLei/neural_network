import tensorflow as tf
import numpy as np
from unit_8_practice import Eighth_total as et

X_train, X_test, Y_train, Y_test, y_train, y_test = et.dataSolve()
# 设置超参数、迭代次数、显示间隔
learn_rate = 0.05
iter = 1000
display_step = 100
# 设置模型参数初始值
np.random.seed(612)
W1 = tf.Variable(np.random.randn(4, 16), dtype=tf.float32)  # 隐含层
B1 = tf.Variable(np.zeros([16]), dtype=tf.float32)
W2 = tf.Variable(np.random.randn(16, 3), dtype=tf.float32)  # 输出层
B2 = tf.Variable(np.zeros([3]), dtype=tf.float32)
# 训练模型
acc_train = []
acc_test = []
cce_train = []
cce_test = []
for i in range(0, iter + 1):
    with tf.GradientTape() as tape:
        Hidden_train = tf.nn.relu(tf.matmul(X_train, W1) + B1)
        PRED_train = tf.nn.softmax(tf.matmul(Hidden_train, W2) + B2)
        Loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train, y_pred=PRED_train))

        Hidden_test = tf.nn.relu(tf.matmul(X_test, W1) + B1)
        PRED_test = tf.nn.softmax(tf.matmul(Hidden_test, W2) + B2)
        Loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test, y_pred=PRED_test))

    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(), axis=1), y_train), tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(), axis=1), y_test), tf.float32))

    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
    cce_train.append(Loss_train)
    cce_test.append(Loss_test)

    grads = tape.gradient(Loss_train, [W1, B1, W2, B2])
    W1.assign_sub(learn_rate * grads[0])
    B1.assign_sub(learn_rate * grads[1])
    W2.assign_sub(learn_rate * grads[2])
    B2.assign_sub(learn_rate * grads[3])
    if i % display_step == 0:
        print("i:%i,TrainAcc:%f,TrainLoss:%f,TestAcc:%f,TestLoss:%f" % (
        i, accuracy_train, Loss_train, accuracy_test, Loss_test))

et.pltShow(acc_train, acc_test, cce_train, cce_test)
