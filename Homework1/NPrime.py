__author__ = 'Mowgli'

from math import sqrt


def prime(n):
    if n < 1:
        print "Please enter a positive number"
        return
    if 1 <= n <= 500:
        size = 4000
    elif 500 < n <= 1000:
        size = 8000
    elif 1000 < n <= 10000:
        size = 150000
    elif 10000 < n <= 100000:
        size = 1300000
    elif 100000 < n <= 1000000:
        size = 16000000
    elif 1000000 < n <= 10000000:
        size = 180000000
    else:
        size = 1000000000

    arr = []
    for p in range(2, size + 1, 1):
        arr.append(1)
    root = int(sqrt(size - 1))

    for j in range(7, root, 1):
        if arr[j] == 1:
            i = j * j
            while i < size - 1:
                arr[i] = 0
                i += 2 * j
    plist = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    i = 41

    while i < size and (i + 26) < size:
        if arr[i] == 1:
            plist.append(i)
        if arr[i + 2] == 1:
            plist.append(i + 2)
        if arr[i + 6] == 1:
            plist.append(i + 6)
        if arr[i + 8] == 1:
            plist.append(i + 8)
        if arr[i + 12] == 1:
            plist.append(i + 12)
        if arr[i + 18] == 1:
            plist.append(i + 18)
        if arr[i + 20] == 1:
            plist.append(i + 20)
        if arr[i + 26] == 1:
            plist.append(i + 26)
        i += 30

    arr[0] = 0
    return plist[n - 1]


def test_run():
    """Driver function called by Test Run."""
    print prime(1)  # should print 2
    print prime(5)  # should print 11


if __name__ == "__main__":
    test_run()
