from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# 下载波士顿房价数据
boston_housing = tf.keras.datasets.boston_housing
# 对数据集进行读取并全部作为训练集
(train_x, train_y), (_, _) = boston_housing.load_data(test_split=0)

# 设置plt的字体和负数表示方法
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置标题
titles = ["CROM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LSTAT", "MENV"]


# 绘制所有属性与房价的关系图
def paintAll():
    plt.figure(figsize=(12, 12))
    for i in range(13):
        plt.subplot(4, 4, (i + 1))
        plt.scatter(train_x[:, i], train_y, label="all")
        plt.xlabel(titles[i])
        plt.ylabel("Price($1000's)")
        plt.title(str(i + 1) + "." + titles[i] + " -Price")

    plt.tight_layout()
    plt.suptitle("各个属性与房价之间的关系", x=0.5, y=1.02, fontsize=20)
    plt.legend()


# 绘制制定的属性与房价的关系图
def paintAppoint(num):
    plt.figure()
    plt.scatter(train_x[:, num - 1], train_y, label="draw=%s" % num)
    plt.xlabel(titles[num])
    plt.ylabel("Price($1000's)")
    plt.title(str(num) + "." + titles[num - 1] + " -price", fontsize=16, color="blue")
    plt.legend()


# 判断输入的是否为数字以及是否在范围内
def numIsCorrect(num):
    if num.isdigit():
        num = int(num)
        if num in range(1, 14):
            return True
        else:
            print("输入的数字范围不正确")
            return False
    else:
        print("输入的不是数字")
        return False


# 菜单函数
def menu():
    for i in range(13):
        print(str(i + 1) + " -- " + titles[i])


# 原本想使用python的__main__标识，但是运行没有办法使用
# __name__ 不能加引号
if __name__ == "__main__":

# 运行函数，相当于main
# def run():
    paintAll()
    i = 1
    menu()
    while i < 4:
        num = input("请选择属性：")
        if numIsCorrect(num):
            num = int(num)
            paintAppoint(num)
            break
        else:
            if i == 3:
                print("已经输入三次，程序结束")
                break
            else:
                print("请重新输入！")
        i += 1
    plt.show()


# run()

# 测试paintAll函数
# paintAll()
# plt.show()
# plt.legend()

# 测试paintAppoint函数
# num = input("请输入一个整数(1-13):")
# num = int(num)
# paintAppoint(num)
# plt.legend()
# plt.show()
