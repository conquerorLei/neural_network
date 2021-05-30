import unit_1_practice.second_bak  # 导入second


# 3-2.py
# range_correct(number):检测数据是不是数值已经是否在范围内
# mulriple(number):返回包含倍数的list数组

# 将结果存入文件
def store():
    time = 0
    # 基础路径
    path = "C://"

    number = input("请输入1-100之间的整数")

    # 调用second_bak中函数
    if second_bak.range_correct(number):

        path = path + number + "的倍数.txt"

        f = open(path, "w+", encoding='utf-8')

        data = second_bak.mulriple(number)

        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')

            s = s.replace("'", '').replace(',', '') + '\n'

            print(s, end="")

            f.writelines(s)

        print("已经存入" + path)

        f.close()

    else:

        print("格式错误，请重新输入！")

        time += 1

        if time < 3:

            return store(time)

        else:

            print("您已经错误输入三次，程序结束")


if __name__ == '__main__':
    store()
