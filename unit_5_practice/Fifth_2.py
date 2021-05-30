import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from unit_5_practice.data import x1
from unit_5_practice.data import x2
from unit_5_practice.data import y

x0 = tf.ones(len(x1), dtype=tf.float32)
X = tf.stack((x0, x1, x2), axis=1)
Y = tf.reshape(y, [-1, 1])


def getW():
    # 转置
    xt = tf.transpose(X)

    # 求逆
    xtx_1 = tf.linalg.inv(tf.matmul(xt, X))
    w = tf.matmul(tf.matmul(xtx_1, xt), Y)

    return tf.reshape(w, -1)


def adjust(number, minimum, maximum):
    if "".join(number.strip("+").strip("-").split(".")).isdigit() and number.count(".") <= 1:
        if number.isdigit():
            number = int(number)
        else:
            number = float(number)
        if minimum <= number <= maximum:
            return True
        else:
            print("不在范围内(", minimum, ",", maximum, ")")
            return False
    else:
        print("输入的不是数字")
        return False


def adjust_1(number):
    string_queue = number.split(".")
    if (number.count(".") == 1 and string_queue[1] == "") or number.count(".") == 0:
        return True
    else:
        print("输入的房间数不是整数")


def imageShow():
    fig = plt.figure(figsize=(8, 6))
    ax3d = Axes3D(fig)

    ax3d.scatter(x1, x2, y, color="blue", marker="*")
    ax3d.set_xlabel('Area', color="red", fontsize=16)
    ax3d.set_ylabel('Room', color='red', fontsize=16)
    ax3d.set_zlabel('Price', color='red', fontsize=16)
    ax3d.set_yticks([1, 2, 3])
    ax3d.set_zlim3d(30, 160)

    plt.show()


if __name__ == "__main__":
    i = 0
    w = getW()
    print(w.numpy())
    while True:
        i += 1
        area = input("请输入面积")
        num = input("请输入房间数")
        if adjust(area, 20, 200) and adjust(num, 1, 10) and adjust_1(num):
            area = float(area)
            num = int(num)
            price = w[0] + w[1] * area + w[2] * num
            print("预测价格：", price.numpy())
            break
        else:
            print("请重新输入")
        if i == 3:
            print("输入三次错误，程序退出")
            break
    imageShow()
