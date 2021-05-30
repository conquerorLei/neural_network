import tensorflow as tf
import numpy as np
from unit_6_practice import total as t
import os

file_name = os.path.basename(__file__)
file_name = file_name.replace(".py", "")

# 下载数据集
boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = t.normalize(train_x, train_y, test_x, test_y)
    mse_train, mse_test, Pred_train, Pred_test, W = t.trainModel(X_train, Y_train, X_test, Y_test, rand=14)
    t.show(mse_train, mse_test, Y_train, Y_test, Pred_train, Pred_test)
    mse_train = np.array([temp.numpy() for temp in mse_train])
    mse_test = np.array([temp.numpy() for temp in mse_test])
    t.outExcel(mse_train, mse_test, filename=file_name)
