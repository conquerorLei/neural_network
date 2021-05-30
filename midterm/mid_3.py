import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from midterm import init as init

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()


def imageShow(image, title):
    """
    :Parameters
    ----------
    image->List[none]
    title->List[none]

    :Returns
    -------
    none

    :Author:  LiXianLei
    :Create:  2021/5/11 22:23
    :Description: # TODO(qingshan110210905@163.com):图片展示
                  # TODO(LiXianLei):变量下标的定义是为了更好的实现所有图片的展示
                  # TODO:在matplotlib源站上对tight_layout()函数的属性rect的解释是元组，为什么带入元组后这个显示警告
    Copyright (c) 2021, LiXianLei Group All Rights Reserved.
    """
    init.Initial.pltInitial()
    plt.figure(figsize=(10, 15))
    index = 0
    for img in image:
        plt.axis("off")  # 关闭坐标轴
        plt.subplot(10, 6, index + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title[index % 6])
        index += 1
    plt.axis("off")
    plt.suptitle("Fashion Mnist数据增强")
    # plt.tight_layout(rect=(0, 0, 1, 1))
    plt.tight_layout()  # 默认情况下rect=（0,0,1,1)
    plt.show()


def imageProcess(image, num, angle=-10):
    """
    :Parameters
    ----------

    :Returns
    -------

    :Author:  LiXianLei
    :Create:  2021/5/12 23:30
    :Description: # TODO(qingshan110210905@163.com):
                  # TODO(LiXianLei):
    Copyright (c) 2021, LiXianLei Group All Rights Reserved.
    """
    if num == 0:
        return np.array(image).reshape((1, 28, 28))  # 原图
    if num == 1:
        return np.array(image.transpose(Image.TRANSPOSE)).reshape((1, 28, 28))  # 转置
    if num == 2:
        return np.array(image.transpose(Image.FLIP_TOP_BOTTOM)).reshape((1, 28, 28))  # 上下翻转
    if num == 3:
        return np.array(image.transpose(Image.FLIP_LEFT_RIGHT)).reshape((1, 28, 28))  # 左右翻转
    if num == 4 or num == 5:
        return np.array(image.rotate(angle)).reshape((1, 28, 28))  # 随机逆时针旋转


def showProperty():
    """
    :Parameters
    ----------

    :Returns
    -------
    none
    :Author:  LiXianLei
    :Create:  2021/5/11 21:11
    :Description: # TODO(qingshan110210905@163.com):输出数据集基本属性，获取数据集所有标签
                  # TODO(LiXianLei):在没有初始化label_set的情况下不能在推导式中使用此变量
    Copyright (c) 2021, LiXianLei Group All Rights Reserved.
    """
    print("Train_x set length:", len(train_x))
    print("Train_x shape:", train_x.shape)
    print("Train_y set length:", len(train_y))
    print("Train_y shape:", train_y.shape)
    print("Test_x set length:", len(test_x))
    print("Test_x shape:", test_x.shape)
    print("Test_y set length:", len(test_x))
    print("Test_y shape:", test_x.shape)
    label_set = {}  # 初始化label_set
    label_set = {x for x in train_y if x not in label_set}  # 利用推导式剔除重复元素，将不重复的所有元素存入label_set
    print("all the labels of fashion_mnist:", label_set)


def imageExtend():
    """
    Parameters
    ----------

    Returns
    -------
    train_x_aug->list[numpy.array]
    :Author:  LiXianLei
    :Create:  2021/5/11 21:50
    :Description: # TODO(qingshan110210905@163.com):拓展前十张图片
                  # TODO(LiXianLei):reshape函数中不能直接写入1,28,28，需要加括号，当做一个元组当做函数参数（即二维以上作为元组传参）
    Copyright (c) 2021, LiXianLei Group All Rights Reserved.
    """
    train_x_aug = train_x[0].reshape((1, 28, 28))
    for i in range(0, 10):
        image = Image.fromarray(train_x[i])
        if i != 0:
            train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 0)))  # 下标不为0则加入
        train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 1)))  # 转置
        train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 2)))  # 上下翻转
        train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 3)))  # 水平翻转
        train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 4)))  # 顺时针旋转10°
        train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 5, 10)))  # 逆时针旋转10°
    return train_x_aug


def imageExtendAllRandom():
    """
    Parameters
    ----------

    Returns
    -------

    :Author:  LiXianLei
    :Create:  2021/5/12 23:30
    :Description: # TODO(qingshan110210905@163.com):
                  # TODO(LiXianLei):
    Copyright (c) 2021, LiXianLei Group All Rights Reserved.
    """
    train_x_aug = train_x[0].reshape((1, 28, 28))  # 初始化train_x_aug
    for i in range(0, 100):
        image = Image.fromarray(train_x[i])
        if i != 0:  # 除0下标之外的其余图像存入
            train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 0)))
        flag = 1  # 次数记录变量
        while flag != 6:  # 随机处理五次
            num = np.random.randint(1, 6, 1)
            angle = np.random.randint(-180, 180)
            train_x_aug = np.concatenate((train_x_aug, imageProcess(image, num, angle)))
            flag += 1
    return train_x_aug


def randomAngle():
    """
    Parameters
    ----------
    self->any

    Returns
    -------
    randomAngle()->train_x_aug:list[numpy.array]

    :Author:  LiXianLei
    :Create:  2021/5/14 11:50
    :Description: # TODO(qingshan110210905@163.com):
                  # TODO(LiXianLei):
    Copyright (c) 2021, LiXianLei Group All Rights Reserved.
    """
    train_x_aug = train_x[0].reshape((1, 28, 28))
    for i in range(0, 10):
        image = Image.fromarray(train_x[i])
        if i != 0:
            train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 0)))
        temp = 1
        while temp != 6:
            angle = np.random.randint(-180, 180)
            train_x_aug = np.concatenate((train_x_aug, imageProcess(image, 4, angle)))
            temp += 1
    return train_x_aug


def randomTenOfHundred(x_aug):
    """
    :Parameters
    ----------
    x_aug:list[numpy.array]

    :Returns
    -------
    temp:list[numpy.array]

    :Author:  LiXianLei
    :Create:  2021/5/14 14:03
    :Description: # TODO(qingshan110210905@163.com):
                  # TODO(LiXianLei):
    Copyright (c) 2021, LiXianLei Group All Rights Reserved.
    """
    num = np.random.randint(0, 100, 10)
    temp = x_aug[6 * num[0]].reshape((1, 28, 28))
    j = 0
    while j < 10:
        if j != 0:
            temp = np.concatenate((temp, x_aug[6 * num[j]].reshape((1, 28, 28))))
        flag = 1
        while flag != 6:
            temp = np.concatenate((temp, x_aug[6 * num[j] + flag].reshape((1, 28, 28))))
            flag += 1
        j += 1
    return temp


if __name__ == "__main__":
    showProperty()
    imageShow(imageExtend(), ['原图', '转置', '上下翻转', '水平翻转', '顺时针', '逆时针'])
    imageShow(randomAngle(), ['原图', '旋转一', '旋转二', '旋转三', '旋转四', '旋转五'])
    imageShow(randomTenOfHundred(imageExtendAllRandom()), ['原图', '随机一', '随机二', '随机三', '随机四', '随机五'])
