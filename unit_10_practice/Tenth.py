import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Sequential

cifar10 = tf.keras.datasets.cifar10
(xc_train, yc_train), (xc_test, yc_test) = cifar10.load_data()
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 数据预处理
Xc_train, Xc_test = tf.cast(xc_train, dtype=tf.float32) / 255.0, tf.cast(xc_test, dtype=tf.float32) / 255.0
Yc_train, Yc_test = tf.cast(yc_train, dtype=tf.int32), tf.cast(yc_test, dtype=tf.int32)
X_train, X_test = tf.cast(train_x, dtype=tf.float32) / 255.0, tf.cast(test_x, dtype=tf.float32) / 255.0
y_train, y_test = tf.cast(train_y, dtype=tf.int32), tf.cast(test_y, dtype=tf.int32)
X_train = train_x.reshape(60000, 28, 28, 1)
X_test = test_x.reshape(10000, 28, 28, 1)

# 建立模型
modelc = Sequential([
    # Unit 1
    layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, input_shape=xc_train.shape[1:]),
    layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2, 2)),
    # Unit 2
    layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),
    layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2, 2)),
    # Unit 3
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model = tf.keras.Sequential([
    # Unit 1
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    # Unit 2
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    # Unit 3
    tf.keras.layers.Flatten(),
    # Unit 4
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 查看摘要
modelc.summary()
model.summary()

# 配置训练方法
modelc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# 训练模型
# batch_size一次训练所选取的样本数,epochs训练模型的次数,validation_split划分验证集
modelc.fit(Xc_train, Yc_train, batch_size=64, epochs=12, validation_split=0.2)
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# 评估模型
# verbose为每个epoch输出一行记录
modelc.evaluate(Xc_test, Yc_test, verbose=2)
model.evaluate(X_test, y_test, verbose=2)
