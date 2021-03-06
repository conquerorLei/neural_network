import tensorflow as tf

x = tf.constant([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
y = tf.constant([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

'''
length = int(tf.size(x))
length = len(x)
'''

w = (int(tf.size(x))*tf.reduce_sum(x * y) - tf.reduce_sum(x) * tf.reduce_sum(y)) / \
    (int(tf.size(x)) * tf.reduce_sum(tf.pow(x, 2)) - tf.pow(tf.reduce_sum(x), 2))
b = (tf.reduce_sum(y) - w * tf.reduce_sum(x))/(int(tf.size(x)))
los = (tf.reduce_sum(w * x + b - y)) / len(x)

print("w = ", w.numpy())
print("b = ", b.numpy())
print("l = ", los.numpy())