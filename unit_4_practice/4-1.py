import tensorflow as tf
import numpy as np

# 创建张量
x = tf.constant([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
y = tf.constant([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

# 求平均值
x_avg = tf.reduce_mean(x, axis=0)
y_avg = tf.reduce_mean(y, axis=0)

# 求w b
w = (tf.reduce_sum((x - x_avg)*(y - y_avg)))/(tf.reduce_sum(tf.pow((x - x_avg),2)))
b = y_avg - w * x_avg
print("w=", w.numpy())
print("b=", b.numpy())