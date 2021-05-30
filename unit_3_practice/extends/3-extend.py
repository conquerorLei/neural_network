import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 下载数据
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

