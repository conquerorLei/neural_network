import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import time  # 导入time数据库
import matplotlib.pyplot as plt

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"  # 下载鸢尾花数据集iris
train_path = tf.keras.utils.get_file("iris_training.csv", TRAIN_URL)
# COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Sepcies']#定义类标题列表，读取数据及文件
df_iris = pd.read_csv(train_path, header=0)  # 将第一行数据作为列标题names=COLUMN_NAMES,
iris = np.array(df_iris)  # 把二维数据表转化成二维numpy数组
train_x = iris[:, 2:4]  # 取花萼的长度和宽度
train_y = iris[:, 4]  # 取最后一列作为标签值
x1_train = train_x[train_y < 1]
x2_train = train_x[train_y > 1]
X_train = np.stack((x1_train, x2_train), axis=0)
x_train = X_train.reshape(-1, 2)
y1_train = train_y[train_y < 1]
y2_train = train_y[train_y > 1]  # 提取山鸢尾与维吉尼亚鸢尾
Y_train = np.stack((y1_train, y2_train), axis=0)
y_train = Y_train.reshape(-1, 1)

num = len(x_train)
x_train = x_train - np.mean(x_train, axis=0)  # 属性中心化
cm_pt = mpl.colors.ListedColormap(['blue', 'green'])
plt.figure(figsize=(5, 3))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)

x0_train = np.ones(num).reshape(-1, 1)
X = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)  # 将某种数据类型的表达式显式转换为另一种数据类型
Y = tf.cast(y_train.reshape(-1, 1), tf.float32)

learn_rate = 0.005  # 设置超参数
iter = 120
display_step = 10
np.random.seed(612)  # 设置模型变量初始值
W = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)  # 设置图变量

x_ = [-2, 3]
y_ = -(W[1] * x_ + W[0]) / W[2]
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)
plt.plot(x_, y_, color="red", linewidth=3)
plt.xlim([-3, 3])
plt.ylim([-3, 3])

start = time.perf_counter()  # 记录训练模型起始时间
ce = []  # 损失
acc = []
for i in range(0, iter + 1):
    with tf.GradientTape() as tape:
        PRED = 1 / (1 + tf.exp(-tf.matmul(X, W)))
        Loss = -tf.reduce_mean(Y * tf.math.log(PRED) + (1 - Y) * tf.math.log(1 - PRED))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED.numpy() < 0.5, 0., 1.), Y), tf.float32))
    ce.append(Loss)
    acc.append(accuracy)

    dL_dW = tape.gradient(Loss, W)
    W.assign_sub(learn_rate * dL_dW)
    if i % display_step == 0:
        print("i: %i,Train Loss: %f,Test Loss: %f" % (i, accuracy, Loss))
        plt.plot(x_, y_, color='r')
end = time.perf_counter()  # 记录训练模型结束时间
print("程序执行时间: ", end - start)
# 可视化输出
plt.figure(figsize=(5, 3))
plt.plot(ce, color='blue', Label='Loss')  # 绘制损失和准确率变化曲线
plt.plot(acc, color='green', Label='acc')
plt.legend()
plt.show()
