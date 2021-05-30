import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 获取数据集
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
# TRAIN_URL.split('/')[-1]表示获取网址中的数据集文件名
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)

# Pandas访问csv数据集
# 制定列标题
COLUMN_NAME = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
# pd读取csv数据
df_iris = pd.read_csv(train_path, names=COLUMN_NAME, header=0)
# 转化成numpy数组
iris = np.array(df_iris)

# 画布设置
fig = plt.figure("Iris Data", figsize=(15, 15))
fig.suptitle("Anderson's Iris Data Set\n(Bule->Setosa | Red->Versicolor | Green->Virginica)", fontsize=20)

# 循环绘制
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 4 * i + (j + 1))
        if i == j:
            plt.hist(iris[:, j], align='mid', histtype='stepfilled', bins=10)
        else:
            plt.scatter(iris[:, j], iris[:, i], c=iris[:, 4], cmap='brg')
        if i == 0:
            plt.title(COLUMN_NAME[j])
        if j == 0:
            plt.ylabel(COLUMN_NAME[i])

plt.show()
