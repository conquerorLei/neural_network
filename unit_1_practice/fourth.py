number = input("请输入1-100的整数:")

i = 0

#基础路径
path = "/home/qs/output_file/"

if number.isdigit():
    #路径补全
    path = path + number + "的倍数.txt"

    number = int(number)

    #打开文件
    f = open(path,"w+")

    #循环控制
    for num in range(1,1001):

        #判断是不是倍数，是倍数则存入文件
        if num % number == 0:

            i = i + 1

            #将结果转化为字符串
            string = str(i) + " " + str(num) + "\n"

            #将结果字符串存入文件
            f.writelines(string)

        #else:
            #有点多余
        #    continue
    #关闭文件
    f.close()

