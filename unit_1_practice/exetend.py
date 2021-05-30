import math
from unit_1_practice import third_bak as tb
a = input("please input a:")
b = input("please input b:")
while not tb.check_correct(a):
    a = input("please input a:")
a = int(a)
while not tb.check_correct(b):
    b = input("please input b:")
b = int(b)
print(int(a*b/math.gcd(a, b)))
