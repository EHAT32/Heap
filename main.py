import numpy as np

class DHeap:
    def __init__(self, alloc_size = 100):
        self.heap = np.zeros(alloc_size) #Первоначальный размер памяти, выделенной под кучу
        self.heap_size = 0

    def siftDown(self, i):
        while 2 * i + 1 < self.heap_size:     # heapSize — количество элементов в куче
            left = 2 * i + 1             # left — левый сын
            right = 2 * i + 2            # right — правый сын
            j = left
            if right < self.heap_size and self.heap[right] < self.heap[left]:
                j = right
            if self.heap[i] <= self.heap[j]:
                break
            self.heap[i], self.heap[j] = self.heap[j], self.heap[i] 
            i = j

    def siftUp(self, i):
        while self.heap[i] < self.heap[int((i - 1) / 2)]:     # i  0 — мы в корне
            self.heap[i], self.heap[int((i - 1) / 2)] = self.heap[int((i - 1) / 2)], self.heap[i]
            i = int((i - 1) / 2)
    
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
    
    @staticmethod
    def merge(heap_1, heap_2):
        while heap_2.heapSize > 0:  
            heap_1.insert(heap_2.extractMin())

def main():
    heap1 = DHeap()
    heap2 = DHeap()

    heap1.insert(4)
    heap2.insert(-4.2)
    heap1.insert(1.9)
    heap2.insert(1e1)

    return

if __name__ == '__main__':
    main()