# linux下的matplotlib字体问题

最近进行了matplotlib的pyplot试用，进行了简单的散点图绘制，但是在进行编译运行的时候发现绘制出来的图像标题和坐标标签文本中的中文字符出现乱码，经过查证是ubuntu系统没有相应的字体造成的

## 解决方案一：

很简单，就是将中文字符换成英文字符，在进行matplotlib字体设置的时候设置ubuntu允许的字体，**但是这是治标不治本的方法**

## 解决方案二：

方案二思路也很简单，“缺啥补啥”是最根本的方法，这样以后在进行中文编辑的时候就没有中文字符编码的问题了。如果想在省事点，直接给他装够常用的中文字体，结束！！！

### 1 复制windows中Fonts文件夹到ubuntu系统fonts中

windows中Fonts位置：`C:/windows/Fonts`

ubuntu中fonts位置：`/usr/fonts`

过程中如果直接复制到ubuntu中fonts中可能会出现问题，可能会提示**权限不够**的错误。解决方法：先复制到普通用户可操作的文件夹，比如桌面。然后在桌面右键，点击“在终端中打开”或者“open in terminal”，在终端中输入以下代码：

```shell
#管理员模式移动文件夹，这样权限就够了
#需要输入密码，这一不应该不用演示了
sudo mv Fonts /usr/share/fonts/
```

### 2 激活移动过来的字体

移动完成后进入在终端中进入`/usr/share/fonts/Fonts/`

```shell
cd /usr/share/fonts/Fonts/
```

输入以下代码,建立字体索引信息，更新字体缓存,让字体生效

```shell
mkfontscale && mkfontdir && fc-cache -fv && source /etc/profile && fc-list |wc -l
```

### *3 仍然可能出现的问题

仍然可能出现的问题就是重新运行设置了matplotlib中文字体的程序绘制的图像中文字符依旧是乱码，这样只需要做的就是**清除matplotlib缓存**，在命令行输入以下代码：

```shell
rm -r ~/.cache/matplotlib
```

如果运行了程序还是有编码问题，重启吧，重启解决80%的问题

## 测试程序：

```python
from matplotlib import pyplot as plt

import numpy as np

plt.rcParams['font.sans-serif'] = "SimHei"

plt.rcParams['axes.unicode_minus'] = False

x = np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])

y = np.array([145.00,110.00,93.00,116.00,65.32,104.00,108.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

plt.scatter(x,y,color="red",marker=".")

plt.title("商品房销售记录",fontsize=16,color="blue")

plt.xlabel("面积(平方米)",fontsize=14)

plt.ylabel("价格(万元)",fontsize=14)

plt.show()
```

