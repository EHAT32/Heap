import numpy as np
from numpy import random

class DHeap:
    def __init__(self, arity = 9, alloc_size = 100, data =  None):
        self.__arity = arity
        if data != None:
            self.__heap = data
            self.__heap_size = np.size(data)
            self.Build_Heap()
        else:
            self.__heap = np.zeros(alloc_size) #Первоначальный размер памяти, выделенной под кучу
            self.__heap_size = 0

    def __len__(self):
        return self.__heap_size

    def siftDown(self, i):
        while self.__arity * i + 1 < self.__heap_size:
            min_child = - 1
            for j in range(self.__arity):
                child = self.__arity * i + j + 1
                if child >= self.__heap_size:
                    break
                if min_child == -1 or self.__heap[min_child] >= self.__heap[child]:
                    min_child, child = child, min_child
            if self.__heap[self.__arity * i] > self.__heap[min_child]:
                self.__heap[self.__arity * i], self.__heap[min_child] = self.__heap[min_child], self.__heap[self.__arity * i]
                i = min_child
            else:
                break
                
        #Найти сына, который меньше текущего элемента
        #Если найдено, меняем местами текущий эл-т и сына
        #Выполнить siftDown для этого сына


    def siftUp(self, i):
        while self.__heap[i] < self.__heap[int((i - 1) / self.__arity)]:
            self.__heap[i], self.__heap[int((i - 1) / self.__arity)] = self.__heap[int((i - 1) / self.__arity)], self.__heap[i]
            i = int((i - 1) / self.__arity)
    
    def extractMin(self):
        min = self.__heap[0]
        self.__heap[0] = self.__heap[self.__heap_size - 1]
        self.__heap_size = self.__heap_size - 1
        self.siftDown(0)
        return min

    def __reallocate_heap(self, new_heap_size):
        if new_heap_size >= np.size(self.__heap):
            heap = np.zeros(np.size(self.__heap) * 2)
            heap[:np.size(heap)] = self.__heap
            self.__heap = heap
    
    def insert(self, key):
        self.__reallocate_heap(self.__heap_size + 1)
        self.__heap_size = self.__heap_size + 1
        self.__heap[self.__heap_size - 1] = key
        self.siftUp(self.__heap_size - 1)

    def Heapify(self, i):
        least = i
        for j in range(self.__arity):
            child = self.__arity * i + j + 1
            #heap_size - количество элементов в куче
            if child >= self.__heap_size:
                break
            if self.__heap[child] < self.__heap[least]:
                least = child
        if least != i:
            self.__heap[i], self.__heap[least] = self.__heap[least], self.__heap[i]
            self.Heapify(least) #self removed

    def Build_Heap(self):
        for i in range(int(self.__heap_size / self.__arity), -1, -1):
            self.Heapify(i)

    @staticmethod
    def merge(a, b):
        for i in  range(b.__heap_size):
            a.__reallocate_heap(a.__heap_size + 1)
            a.__heap_size = a.__heap_size + 1
            a.__heap[a.__heap_size - 1] = b.__heap[i]
        a.Build_Heap()

def Dijkstra(Graph, source):
    vertex_count = np.size(Graph, 0)
    Q = list(range(vertex_count))
    dist = [None] * vertex_count
    prev = [None] * vertex_count                 
    dist[source] = 0                       
     
    while Q:
        min_idx = np.argmin(Q)
        u = Q[min_idx]   
                                            
        Q.pop(min_idx)
        
        for i in range(vertex_count):
            if Graph[u, i] == None:
                continue
            alt = dist[u] + Graph[u, i]
            if dist[i] == None or alt < dist[i]:              
                dist[i] = alt
                prev[i] = u

    return dist, prev

def Dijkstra_heap(Graph, source):
    vertex_count = np.size(Graph, 0)
    Q = DHeap(data = list(range(vertex_count)))
    dist = [None] * vertex_count
    prev = [None] * vertex_count                 
    dist[source] = 0                       
     
    while Q:
        u = Q.extractMin()   
        for i in range(vertex_count):
            if Graph[u, i] == None:
                continue
            alt = dist[u] + Graph[u, i]
            if dist[i] == None or alt < dist[i]:              
                dist[i] = alt
                prev[i] = u

    return dist, prev

def main():
    Graph = np.array(
        [
            [0   , 7   , 9   , None, None, 14  ], 
            [7   , 0   , 10  , 15  , None, None],
            [9   , 10  , 0   , 11  , None, 2   ],
            [None, 15  , 11  , 0   , 6   , None],
            [None, None, None, 6   , 0   , 9   ],
            [14  , None, 2   , None, 9   , 0   ]
        ]
    )

    #dist, prev = Dijkstra(Graph, 0)

    dist, prev = Dijkstra_heap(Graph, 0)

    return

if __name__ == '__main__':
    main()