import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from midterm import init as init
import time

init.Initial.pltInitial()

boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

# 获取低收入人口比例训练集
x_train = train_x[:, 12]
y_train = train_y
x_test = test_x[:, 12]
y_test = test_y

# 设置超参数
# 超参数的设置是至关重要的
learn_rate = 0.009
my_iter = 2000
display_step = 200

# 设置模型初始值
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# 训练集损失
mse_train = []
# 测试机损失
mse_test = []

start_time = time.time()
for i in range(1, my_iter + 1):
    with tf.GradientTape() as tape:
        # 预处理指定的训练集和数据集
        pred_train = w * x_train + b
        loss_train = 0.5*tf.reduce_mean(tf.square(y_train - pred_train))

        pred_test = w * x_test + b
        loss_test = 0.5*tf.reduce_mean(tf.square(y_test - pred_test))

    # 将本次训练均方差损失的值存入对应的数组
    mse_train.append(loss_train)
    mse_test.append(loss_test)

    dL_dw, dL_db = tape.gradient(loss_train, [w, b])
    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    if i % display_step == 0:
        print("i: %i,Train Loss: %f, Test Loss: %f" % (i, loss_train, loss_test))

end_time = time.time()
print("time cast: %fs" % (end_time - start_time))

plt.figure(figsize=(15, 10))

plt.subplot(221)
plt.scatter(x_train, y_train, color="blue", label="data")
plt.plot(x_train, pred_train, color="red", label="model")
plt.legend(loc="upper left")

plt.subplot(222)
plt.plot(mse_train, color="blue", linewidth=3, label="train loss")
plt.plot(mse_test, color="red", linewidth=3, label="test loss")
plt.legend(loc="upper right")

plt.subplot(223)
plt.plot(y_train, color="blue", marker="o", label="true price")
plt.plot(pred_train, color="red", marker=".", label="predict")
plt.legend()

plt.subplot(224)
plt.plot(y_test, color="blue", marker="o", label="true price")
plt.plot(pred_test, color="red", marker=".", label="predict")
plt.legend()

plt.show()
