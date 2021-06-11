import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文黑体
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 6))

# 1、加载数据
# 下载鸢尾花数据集iris
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file("iris_training.csv", TRAIN_URL)
df_iris = pd.read_csv(train_path, header=0)  # pandas读取csv文件，将第一行数据作为列标题names=COLUMN_NAMES

# 2、处理数据
iris = np.array(df_iris)  # 把二维数据表转化成二维numpy数组
# 提取属性和标签
train_x = iris[:, 2:4]  # 取花瓣的长度和宽度
train_y = iris[:, 4]  # 取最后一列作为标签值
# 提取变色鸢尾和维吉尼亚鸢尾
x_train = train_x[train_y > 0]
y_train = train_y[train_y > 0]
num = len(x_train)  # 记录现在的样本数

# 对每个属性按列进行中心化，中心化后样本被整体平移，横纵坐标均值为0
x_train = x_train - np.mean(x_train, axis=0)

# 生成多元模型的属性矩阵和标签列向量
x0_train = np.ones(num).reshape(-1, 1)
X = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)
Y = tf.one_hot(tf.constant(y_train, dtype=tf.int32), 3)

# 3、设置超参数
learn_rate = 0.9
iter = 200

display_step = 20

# 4、设置模型变量初始值
np.random.seed(612)
W = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)

start = time.perf_counter()  # 记录训练模型起始时间

# 5、训练模型，训练过程中绘制决策边界
x_ = [-3, 3]
y_ = -(W[1] * x_ + W[0]) / W[2]
cm_pt = mpl.colors.ListedColormap(['red', 'green'])
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)  # 开始训练模型之前绘制散点图
plt.title("变色鸢尾和维吉尼亚鸢尾散点图和决策边界", fontsize=14)
# 使用模型参数初始值绘制决策边界对应的直线
plt.plot(x_, y_, color='blue', linewidth=3)
plt.xlim([-4, 4])
plt.ylim([-2, 2])

cce = []  # 保存每次迭代的交叉熵损失
acc = []  # 保存准确率
for i in range(0, iter + 1):
    with tf.GradientTape() as tape:
        PRED = 1 / (1 + tf.exp(-tf.matmul(X, W)))
        Loss = -tf.reduce_mean(Y * tf.math.log(PRED) + (1 - Y) * tf.math.log(1 - PRED))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED.numpy() < 0.5, 0., 1.), Y), tf.float32))
    cce.append(Loss)
    acc.append(accuracy)
    # 更新模型参数
    dL_dW = tape.gradient(Loss, W)
    W.assign_sub(learn_rate * dL_dW)

    if i % display_step == 0:
        print("i: %i,Train Accuracy: %f,Train Loss: %f" % (i, accuracy, Loss))
        # 每次输出结果时，使用当前模型参数计算纵坐标，并根据其绘制分类直线
        plt.plot(x_, y_, color='blue')

end = time.perf_counter()  # 记录训练模型结束时间
print("程序执行时间: ", end - start)
plt.figure(figsize=(5, 3))
plt.plot(cce, color='blue', Label='Loss')  # 绘制损失和准确率变化曲线
plt.plot(acc, color='green', Label='acc')
plt.legend()
plt.show()
