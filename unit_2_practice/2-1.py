import numpy as np

np.random.seed(612)

list = np.random.random(size=1001)

num = input("please input an Integer")

if num.isdigit():

    num = int(num)

    if num > 0 and num <= 100 :

        j = 1 ;

        print("serial index random")

        for i in range(0,1001,num) :

            print(j,i,list[i])

            j += 1

    else:

        print("the number you input is not in range(0,1000)")

else:

    print("the number you input is not digit")
