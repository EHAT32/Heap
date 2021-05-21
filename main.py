import numpy as np
from numpy import random

class DHeap:
    def __init__(self, alloc_size = 100):
        self.heap = np.zeros(alloc_size) #Первоначальный размер памяти, выделенной под кучу
        self.heap_size = 0
        self.arity = 5

    def siftDown(self, i):
        while self.arity * i + 1 < self.heap_size:
            min_child = - 1
            for j in range(self.arity):
                child = self.arity * i + j + 1
                if child >= self.heap_size:
                    break
                if min_child == -1 or self.heap[min_child] >= self.heap[child]:
                    min_child, child = child, min_child
            if self.heap[self.arity * i] > self.heap[min_child]:
                self.heap[self.arity * i], self.heap[min_child] = self.heap[min_child], self.heap[self.arity * i]
                i = min_child
            else:
                break
                
        #Найти сына, который меньше текущего элемента
        #Если найдено, меняем местами текущий эл-т и сына
        #Выполнить siftDown для этого сына


    def siftUp(self, i):
        while self.heap[i] < self.heap[int((i - 1) / self.arity)]:
            self.heap[i], self.heap[int((i - 1) / self.arity)] = self.heap[int((i - 1) / self.arity)], self.heap[i]
            i = int((i - 1) / self.arity)
    
    def extractMin(self):
        min = self.heap[0]
        self.heap[0] = self.heap[self.heap_size - 1]
        self.heap_size = self.heap_size - 1
        self.siftDown(0)
        return min

    def __reallocate_heap(self, new_heap_size):
        if new_heap_size >= np.size(self.heap):
            heap = np.zeros(np.size(self.heap) * 2)
            heap[:np.size(heap)] = self.heap
            self.heap = heap
    
    def insert(self, key):
        self.__reallocate_heap(self.heap_size + 1)
        self.heap_size = self.heap_size + 1
        self.heap[self.heap_size - 1] = key
        self.siftUp(self.heap_size - 1)

    def Heapify(self, i):
        least = i
        for j in range(self.arity):
            child = self.arity * i + j + 1
            #heap_size - количество элементов в куче
            if child >= self.heap_size:
                break
            if self.heap[child] < self.heap[least]:
                least = child
        if least != i:
            self.heap[i], self.heap[least] = self.heap[least], self.heap[i]
            self.Heapify(least) #self removed

    def Build_Heap(self):
        for i in range(int(self.heap_size / self.arity), -1, -1):
            self.Heapify(i)

    
    
    @staticmethod
    def merge(a, b):
        for i in  range(b.heap_size):
            a.__reallocate_heap(a.heap_size + 1)
            a.heap_size = a.heap_size + 1
            a.heap[a.heap_size - 1] = b.heap[i]
        a.Build_Heap()

def main():
    heap1 = DHeap()
    heap2 = DHeap()
    
    heap1.insert(3)
    heap2.insert(-5)
    heap2.insert(10)
    heap1.insert(1)
    heap1.insert(9)
    heap2.insert(0.5)
    heap2.insert(8)
    heap1.insert(-2)


    DHeap.merge(heap1, heap2)

    return

if __name__ == '__main__':
    main()