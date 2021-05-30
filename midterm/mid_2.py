import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import midterm.init as init

init.Initial.pltInitial()


class NumberImage:
    def __init__(self, path="images/test2/", extension=".png"):
        self.path = path
        self.extension = extension

    def setPath(self, path):
        self.path = path

    def setExtension(self, extension):
        self.extension = extension

    def getImagePathList(self):
        """
        Parameters
        ----------
        self:none

        Returns
        -------
        list which contains all images which extension is specified

        :Author:  LiXianLei
        :Create:  2021/5/7 21:55
        :Description: # TODO(qingshan110210905@163.com):返回图片路径列表
        Copyright (c) 2021, LiXianLei Group All Rights Reserved.
        """
        return [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(self.extension)]

    def imageProcessing(self):
        """
        Parameters
        ----------
        self:none

        Returns
        -------
        the list which contains all the processed images,the type of the list element is numpy.array

        :Author:  LiXianLei
        :Create:  2021/5/11 15:07
        :Description: # TODO(qingshan110210905@163.com):图片处理（读取，二值化，反置，转化为numpy array)
        Copyright (c) 2021, LiXianLei Group All Rights Reserved.
        """
        img_list = []
        img_path = self.getImagePathList()
        i = 0
        for path in img_path:
            img = Image.open(path)  # 打开图片
            img_gray = img.convert("1")  # 转化为二值图像
            img_gray_array = np.array(img_gray)  # 蒋土祥转化为numpy数组
            img_invert_array = np.invert(img_gray_array)  # 反置运算 0->1 1->0
            img_invert = Image.fromarray(img_invert_array)  # 将numpy数组转为图像
            img_small = img_invert.resize((28, 28))  # 将图像缩放为28*28
            plt.imshow(img_small)
            img_small.save(self.path + "process/no" + str(i) + self.extension)  # 将图像按照原路径保存
            img_small_array = np.array(img_small).reshape(28, 28)  # 将缩放后的图像转化为numpy数组
            img_list.append(img_small_array)  # 将数组存入图片数组中
        return img_list

    def typedDataSet(self):
        """
        Parameters
        ----------
        self:any
        Returns
        -------
        (train_x,train_y)
        :Author:  LiXianLei
        :Create:  2021/5/11 15:52
        :Description: # TODO(qingshan110210905@163.com):使用concatenate函数和append函数失效，无法拼接，原因未知
        Copyright (c) 2021, LiXianLei Group All Rights Reserved.
        """
        img_list = self.imageProcessing()
        train_x = np.array(img_list)
        train_y = np.arange(10)
        return train_x, train_y

    def imageShow(self):
        """
        Parameters
        ----------
        self:none

        Returns
        -------
        none

        :Author:  LiXianLei
        :Create:  2021/5/11 15:57
        :Description: # TODO(qingshan110210905@163.com):展示处理后的图像
        Copyright (c) 2021, LiXianLei Group All Rights Reserved.
        """
        train_x, train_y = self.typedDataSet()
        for i in range(0, 10):
            plt.subplot(2, 5, i + 1)
            plt.axis("off")
            plt.imshow(train_x[i], cmap="gray")
            plt.title(train_y[i])
        plt.show()


if __name__ == "__main__":
    ni = NumberImage()
    ni.imageShow()
