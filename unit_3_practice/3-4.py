from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# 设置字体及其样式
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 下载数据集
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 创建画布
plt.figure()
# 设置全局标题及其样式
plt.suptitle("MNIST测试集样本", fontsize=20, color="red")

# 循环绘制子图
for i in range(16):
    num = np.random.randint(1,10000)

    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(test_x[num], cmap="gray")
    plt.title("标签值："+str(test_y[num]), fontsize=14)

plt.show()
