def check_correct(number):
    if number.isdigit():

        print("输入的整数是:" + number)

        number = int(number)

        if number in range(1, 101):

            return True

        else:

            print("您输入的数字的范围不正确，请重新输入")

    else:

        print("您的输入无效，请重新输入")

    return False


def println(number):
    i = 0

    number = int(number)

    for num in range(1, 1001):

        if num % number == 0:
            i += 1

            print(i, num)


if __name__ == '__main__':
    for j in range(1, 4):

        number = input("请输入1-100之间的整数")

        if check_correct(number):

            j = j - 1
            print("输入整数为："+number)
            #
            # println(number)

            break

    if j == 3:
        print("您已经三次输入错误，程序退出")
