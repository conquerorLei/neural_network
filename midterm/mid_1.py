import matplotlib.pyplot as plt
import numpy as np
import midterm.init as init


if __name__ == "__main__":
    # 初始化画布
    init.Initial.pltInitial()
    # x取值
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    # 获取sin,cos
    sin, cos = np.sin(x), np.cos(x)

    # 设置标题
    plt.title('正/余弦函数图像', fontsize=16, color='black')
    # 绘图
    plt.plot(x, sin, color='blue', linewidth=2.5, label='正弦')
    plt.plot(x, cos, color='red', linewidth=2.5, label='余弦')
    # 设置横纵坐标标签
    plt.xlabel('x轴', fontsize=12)
    plt.ylabel('y轴', fontsize=12)

    # 拉伸1.5倍
    plt.xlim(min(x) * 1.5, max(x) * 1.5)
    plt.ylim(min(sin) * 1.5, max(sin) * 1.5)
    '''
    :function 设置横坐标并更换以特殊字符pi
    :parameter ticks 刻度
    :parameter label 标签
    :description 为了使画出的图像更加符合我们平常学习工作的习惯，将数字刻度代替为pi
    '''
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.yticks([-1, 0, 1])
    '''
    question: 尽管x取值没问题，但是在作图时会出现图像左右有空白区域的情况，这个时候不能使用移动左右边框来实现
            这样会出现很奇怪的现象，上下边框没有自动缩减出现罗马数字2的奇怪现象
    solution: 设置x轴范围而不是移动轴，不要那么暴力
    '''
    # ax = plt.gca()
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data', -np.pi))
    # ax.spines['right'].set_position(('data', np.pi))
    plt.gca().set_xlim(-np.pi, np.pi)
    plt.legend()
    plt.show()
