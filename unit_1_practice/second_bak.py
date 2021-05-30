def range_correct(number):
    if number.isdigit():
        i = int(number)
        if i in range(1, 100):
            return True
    return False


# if __name__ == '__main__':
#     while True:
#         a = input("请输入一个整数")
#         if a.isdigit():
#             print("您输入的是整数：" + a)
#             break
#         else:
#             print("请重新输入")


def mulriple(number):
    list = []
    i = 0
    if range_correct(number):
        number = int(number)
        for num in range(1, 1001):
            if num % number == 0:
                i = i + 1
                list.append([i, num])
    # print(list)
    return list
