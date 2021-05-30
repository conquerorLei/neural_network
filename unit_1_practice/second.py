number = input("please input an integer:")

i = 0
if number.isdigit():
    number = int(number)
    for num in range(1, 1001):
        if num % number == 0:
            i = i + 1
            print(i, num)
