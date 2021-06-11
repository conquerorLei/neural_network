import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
type(train_x), type(train_y)
type(test_x), type(test_y)
# 数据预处理
# X_train=train_x.reshape((60000,28*28))
# X_test=test_x.reshape((10000,28*28))
tf.keras.layers.Flatten()
# print(X_train.shape)
# print(X_test.shape)
# cast()函数数据转换
X_train, X_test = tf.cast(train_x / 255.0, tf.float32), tf.cast(test_x / 255.0, tf.float32)
y_train, y_test = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)

type(X_train), type(y_train)
# 建立模型
model = tf.keras.Sequential()  # 不计算，进行形状转换
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.summary()
# 配置训练方法
model.compile(optimizer='adam',  # 优化器
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
# 训练模型
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# 评估模型
model.evaluate(X_test, y_test, verbose=2)

# 使用模型
plt.axis("off")
plt.imshow(test_x[0], cmap="gray")
plt.show()

# 使用模型-测试集中前四个数据
# model.predict([[X_test[0]]])  # 识别图片
# np.argmax(model.predict([[X_test[0]]]))

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.axis("off")
    plt.imshow(test_x[i], cmap='gray')  # 前四个值变成一维依次输出
    plt.title(test_y[i])
plt.show()

# model.predict(X_test[0:4])

y_pred = np.argmax(model.predict(X_test[0:4]), axis=1)
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.axis("off")
    plt.imshow(test_x[i], cmap='gray')
    plt.title("y=" + str(test_y[i]) + "\np_pred" + str(y_pred[i]))  # 标签值和预测值
plt.show()

# 测试集中随机取4个数据
for i in range(4):
    num = np.random.randint(1, 10000)
    plt.subplot(1, 4, i + 1)
    plt.axis("off")
    plt.imshow(test_x[num], cmap='gray')
    y_pred = np.argmax(model.predict(X_test[num], batch_size=32))
    title = "y=" + str(test_y[num]) + "\np_pred" + str(y_pred)
    plt.title(title)
plt.show()
