
while True:
    i = input("please input an intrger:")

    j = 0

    if i.isdigit():
        print("您输入的是整数：",i)
        i = int(i)
        for num in range(1,1001):
            if num % i == 0:
                j = j + 1
                print(j,num)
        break
    else:
        print("您输入的不是整数，请重新输入")
