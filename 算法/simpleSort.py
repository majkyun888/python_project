# -*- encoding = utf-8 -*-


def swap(arr, i ,j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp


def insertSort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0:
            if (arr[j] < arr[j-1]):
                swap(arr, j, j -1)
            j -= 1
    return arr


def insertSort2(arr):
    for i in range(1, len(arr)):
        temp = arr[i]
        j = i
        while j > 0:
            if (arr[j - 1] > temp):
                arr[j] = arr[j - 1]
            else:
                break
            j -= 1
        arr[j] = temp
    return arr


def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if (arr[j] < arr[i]):
                swap(arr, i, j)
    return arr


def selectSort(arr):
    for i in range(len(arr)):
        min = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[i]:
                min = j
        swap(arr, i, min)
    return arr


if __name__ == '__main__':

    arr = [10 , 8, 6, 5, 4, 2,1, 0]
    arr = selectSort(arr)
    for i in arr:
        print(i)