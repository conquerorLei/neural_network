import tensorflow as tf
from unit_5_practice.data import x1 as x
from unit_5_practice.data import y

# 求解权值以及偏置值
w = tf.reduce_sum((x - tf.reduce_mean(x)) * (y - tf.reduce_mean(y))) / \
    tf.reduce_sum(tf.square(x - tf.reduce_mean(x)))
b = tf.reduce_mean(y) - w * tf.reduce_mean(x)

print("权值=", w.numpy(), "\t偏置值=", b.numpy())
print("线性模型：y=", w.numpy(), "* x +", b.numpy())

if __name__ == "__main__":
    i = 0
    while True:
        area = input("请输入面积：")
        if "".join(area.strip("+").strip("-").split(".")).isdigit():
            area = float(area)
            if 20 < area < 200:
                prediction = area * w + b
                print("预测值：" + str(prediction.numpy()))
                break
            else:
                print("输入数据范围不正确！请重新输入")
        else:
            print("您输入的不是数字！请重新输入")
        i += 1
        if i == 3:
            print("您已经三次输入错误，程序退出")
