import tensorflow as tf
import numpy as np
import unit_6_practice.total as t
import os

file_name = os.path.basename(__file__)
file_name = file_name.replace(".py", "")

# 下载数据集
boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()
index = [4, 5, 12]
train_x = np.array([train_x[:, i] for i in index]).T
test_x = np.array([test_x[:, i] for i in index]).T

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = t.normalize(train_x, train_y, test_x, test_y)
    mse_train, mse_test, Pred_train, Pred_test, W = t.trainModel(X_train, Y_train, X_test, Y_test, rand=4)
    t.show(mse_train, mse_test, train_y, test_y, Pred_train, Pred_test)
    mse_train = np.array([temp.numpy() for temp in mse_train])
    mse_test = np.array([temp.numpy() for temp in mse_test])
    t.outExcel(mse_train, mse_test, filename=file_name)

# train_c = train_x[:, 4:6]
# temp = train_x[:, 12]
# temp = temp[:, np.newaxis]
# train_c = np.hstack([train_c, temp])
#
# test_c = test_x[:, 4:6]
# temp = test_x[:, 12]
# temp = temp[:, np.newaxis]
# test_c = np.hstack([test_c, temp])
# print(train_c.shape)
# 归一化处理
# x_train = (train_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
# # y_train = np.array([train_y[i] for i in index])
# y_train = train_y
#
# x_test = (test_x - test_x.min(axis=0)) / (test_x.max(axis=0) - test_x.min(axis=0))
# # y_test = np.array([test_y[i] for i in index])
# y_test = test_y
#
# # 堆叠
# x0_train = np.ones(len(train_x)).reshape((-1, 1))
# x0_test = np.ones(len(test_x)).reshape((-1, 1))
#
# # 转化为张量
# X_train = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)
# X_test = tf.cast(tf.concat([x0_test, x_test], axis=1), tf.float32)
#
# Y_train = tf.constant(y_train.reshape((-1, 1)), tf.float32)
# Y_test = tf.constant(y_test.reshape((-1, 1)), tf.float32)
#
# # 设置超参数
# learn_rate = 0.01
# my_iter = 2000
# display_step = 200
#
# # 设置模型变量初始值
# np.random.seed(612)
# W = tf.Variable(np.random.randn(4, 1), dtype=tf.float32)
#
# # 训练模型
# mse_train = []
# mse_test = []
#
# for i in range(0, my_iter+1):
#     with tf.GradientTape() as tape:
#         Pred_train = tf.matmul(X_train, W)
#         Loss_train = 0.5 * tf.reduce_mean(tf.square(Y_train - Pred_train))
#
#         Pred_test = tf.matmul(X_test, W)
#         Loss_test = 0.5 * tf.reduce_mean(tf.square(Y_test - Pred_test))
#
#     mse_train.append(Loss_train)
#     mse_test.append(Loss_test)
#
#     dL_dW = tape.gradient(Loss_train, W)
#     W.assign_sub(learn_rate * dL_dW)
#
#     if i % display_step == 0:
#         print("i: %i, Train Loss: %f, Test Loss: %f" % (i, Loss_train, Loss_test))
#
# # 可视化输出
# plt.figure(figsize=(20, 4))
#
# plt.subplot(131)
# plt.ylabel("MSE")
# plt.plot(mse_train, color="blue", linewidth=3)
# plt.plot(mse_test, color="red", linewidth=1.5)
#
# plt.subplot(132)
# plt.ylabel("Price")
# plt.plot(y_train, color="blue", marker="o", label="true_price")
# plt.plot(Pred_train, color="red", marker=".", label="pred_price")
# plt.legend()
#
# plt.subplot(133)
# plt.ylabel("Price")
# plt.plot(y_test, color="blue", marker="o", label="true_price")
# plt.plot(Pred_test, color="red", marker=".", label="pred_price")
# plt.legend()
#
# plt.show()
