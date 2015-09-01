__author__ = 'Mowgli'

from math import sqrt
import time

s = time.time()


def prime(n):
    n += 1
    arr = []
    for p in range(2, n + 1, 1):
        arr.append(1)

    root = int(sqrt(n))

    for i in range(2, root + 1, 1):
        if arr[i] == 1:
            for j in range(i * i, n, i):
                arr[j - 1] = 0

    arr[0] = 0
    #print arr


prime(10000000)
print time.time() - s
# s = time.time()
# num = 1
# pcnt = 0
#
#
# def prime(n):
#     sqroot = int(sqrt(n))
#     j = 2
#     while j <= sqroot:
#         if n % j == 0:
#             return False
#         j += 1
#     return True
#
#
# while (1):
#     num += 1
#     if prime(num):
#         pcnt += 1
#     if pcnt == 1000000:
#         print pcnt, 'th prime is', num
#         break
#
# print time.time() - s
