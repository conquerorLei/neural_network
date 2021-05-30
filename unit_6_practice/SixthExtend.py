import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from midterm import init as init
from unit_6_practice import total as t
from unit_5_practice import Fifth_2 as f2

file_name = os.path.basename(__file__)
file_name = file_name.replace(".py", "")

init.Initial.pltInitial()

# pandas 读取数据
df = pd.read_excel("../resource/HousePrices.xlsx")
house_prices = df.values
# 删除nan列
house_prices = np.delete(house_prices, 3, axis=1)
# 切片,获取x和y
x = house_prices[:, 0:2]
y = np.array(house_prices[:, 2]).T
# 获取train_x,train_y,test_x,test_y,比例4:1
train_x = house_prices[0:80, 0:2]
test_x = np.array(house_prices[80:100, 0:2])
train_y = np.array(house_prices[0:80, 2]).T
test_y = np.array(house_prices[80:100, 2]).T


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = t.normalize(train_x, train_y, test_x, test_y)
    mse_train, mse_test, Pred_train, Pred_test, W = t.trainModel(X_train, Y_train, X_test, Y_test, rand=3, rate=0.007, miter=4000)
    t.show(mse_train, mse_test, train_y, test_y, Pred_train, Pred_test)
    mse_train = np.array([temp.numpy() for temp in mse_train])
    mse_test = np.array([temp.numpy() for temp in mse_test])
    t.outExcel(mse_train, mse_test, filename=file_name)
    W = W.numpy()
    # print(W)
    # print(type(W))
    fig = plt.figure(figsize=(8, 6))
    ax3d = Axes3D(fig)

    ax3d.scatter(train_x[:, 0], train_x[:, 1], train_y, color="blue", marker="*")
    x0, x1 = np.meshgrid(train_x[:, 0], train_x[:, 1])
    z = W[0] + x0 * W[1] + x1 * W[2]
    ax3d.plot_surface(x0, x1, z)
    ax3d.set_xlabel('Area', color="red", fontsize=16)
    ax3d.set_ylabel('Year', color='red', fontsize=16)
    ax3d.set_zlabel('Price', color='red', fontsize=16)
    ax3d.set_zlim3d(30, 160)
    plt.show()
    i = 0
    while True:
        i += 1
        area = input("请输入面积:")
        year = input("请输入年份:")
        if f2.adjust(area, 20, 200) and f2.adjust(year, 1900, 3000):
            area = float(area)
            year = int(year)
            price = W[0] + W[1] * area + W[2] * year
            print(type(price))
            print("预测价格：", price)
            break
        else:
            print("请重新输入")
        if i == 3:
            print("输入三次错误，程序退出")
            break
