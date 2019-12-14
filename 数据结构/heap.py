# -*- encoding = utf-8 -*-


class MaxHeap(object):

    def __init__(self, hp=None):
        if not hp:
            self.size = 0
            self.heap = []
        else:
            self.heap = hp
            self.size = len(hp)
            i = self.size - 1
            while i:
                self._siftDown(self.__parent(i))
                i -= 1

    def _swap(self,i, j):
        temp = self.heap[i]
        self.heap[i] = self.heap[j]
        self.heap[j] = temp

    def __parent(self, k):
        return int((k - 1) / 2)

    def __leftChild(self, k):
        return int((2*k) + 1)

    def __rightChild(self, k):
        return int((2*k) + 2)

    def __siftUp(self, k):
        while k > 0 and self.__parent(k) < self.heap[k]:
            self._swap(self.__parent(k), k)
            k = self.__parent(k)

    def _siftDown(self, k):
        while self.__leftChild(k) < self.size:
            j = self.__leftChild(k)
            if j + 1 < self.size and self.heap[j] < self.heap[j + 1]:
                    j += 1
            if self.heap[k] >= self.heap[j]:
                break
            self._swap(k, j)
            k = j

    def size(self):
        return self.size

    def isEmpty(self):
        return self.size == 0

    def insert(self, item):
        self.heap.append(item)
        self.size += 1
        self.__siftUp(self.size - 1)

    def extractMax(self):
        ret = self.heap[0]
        self._swap(0, self.size - 1)
        self.heap.pop()
        self.size -= 1
        self._siftDown(0)
        return ret

