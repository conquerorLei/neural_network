# 求回归方程

import numpy as np

x = [64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]

y = [62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

x_size = np.size(x)

y_size = np.size(y)

x_average = np.mean(x)

y_average = np.mean(y)

if x_size == y_size:

    size = x_size

else:

    size = 0


def denominator():
    temp = 0

    for i in range(0, size):
        temp += (x[i] - x_average) ** 2

    return temp


def numerator():
    temp = 0

    for i in range(0, size):
        temp += (x[i] - x_average) * (y[i] - y_average)

    return temp


if __name__ == "__main__":
    w = numerator() / denominator()

    b = y_average - w * x_average

    print(w, b)
