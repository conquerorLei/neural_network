import matplotlib.pyplot as plt


class Initial(object):
    @staticmethod
    def pltInitial():
        """
        Parameters
        ----------

        Returns
        -------

        :Author:  LiXianLei
        :Create:  2021/5/6 22:29
        :Description: # TODO(qingshan11020905@163.com):
                      # TODO(LiXianLei):
        Copyright (c) 2021, LiXianLei Group All Rights Reserved.
        """
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置字体
        plt.rcParams['axes.unicode_minus'] = False  # 设置负号显示格式

    @staticmethod
    def readImage(path):
        """
        Parameters
        ----------

        Returns
        -------

        :Author:  LiXianLei
        :Create:  2021/5/7 21:28
        :Description: # TODO(qingshan110210905@163.com):完成对文件夹内图像文件的读取
                      # TODO(LiXianLei):
        Copyright (c) 2021, LiXianLei Group All Rights Reserved.
        """
