import numpy as np
from numpy import random
import time

class DHeap:
    def __init__(self, arity = 9, alloc_size = 100, dtype = int):
        self.__arity = arity
        self.__heap = np.zeros(alloc_size) #Первоначальный размер памяти, выделенной под кучу
        self.__heap_size = 0
        self.__values = np.zeros(alloc_size, dtype = dtype)
        self.__dtype = dtype

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
                self.__values[self.__arity * i], self.__values[min_child] = self.__values[min_child], self.__values[self.__arity * i]
                i = min_child
            else:
                break
                
        #Найти сына, который меньше текущего элемента
        #Если найдено, меняем местами текущий эл-т и сына
        #Выполнить siftDown для этого сына


    def siftUp(self, i):
        while self.__heap[i] < self.__heap[int((i - 1) / self.__arity)]:
            self.__heap[i], self.__heap[int((i - 1) / self.__arity)] = self.__heap[int((i - 1) / self.__arity)], self.__heap[i]
            self.__values[i], self.__values[int((i - 1) / self.__arity)] = self.__values[int((i - 1) / self.__arity)], self.__values[i]
            i = int((i - 1) / self.__arity)
    
    def extractMin(self):
        min_key = self.__heap[0]
        min_value = self.__values[0]
        self.__heap[0] = self.__heap[self.__heap_size - 1]
        self.__values[0] = self.__values[self.__heap_size - 1]
        self.__heap_size = self.__heap_size - 1
        self.siftDown(0)
        return min_key, min_value

    def __reallocate_heap(self, new_heap_size):
        if new_heap_size >= np.size(self.__heap):
            heap = np.zeros(np.size(self.__heap) * 2)
            heap[:np.size(heap)] = self.__heap
            self.__heap = heap
            values = np.zeros(np.size(self.__values) * 2, dtype = self.__dtype)
            values[:np.size(values)] = self.__values
            self.__values = values
    
    def insert(self, key, value):
        self.__reallocate_heap(self.__heap_size + 1)
        self.__heap_size = self.__heap_size + 1
        self.__heap[self.__heap_size - 1] = key
        self.__values[self.__heap_size - 1] = value
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
            self.__values[i], self.__values[least] = self.__values[least], self.__values[i]
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
            a.__values[a.__heap_size - 1] = b.__values[i]
        a.Build_Heap()

    @staticmethod
    def make_dheap(keys, values, arity = 9):
        dheap = DHeap(arity, alloc_size = 0)
        dheap.__heap_size = np.size(keys)
        dheap.__heap = np.copy(keys)
        dheap.__values = np.copy(values)
        dheap.__arity = arity
        dheap.Build_Heap()
        return dheap


def Dijkstra(Graph, source):
    vertex_count = np.size(Graph, 0)
    Q = list(range(vertex_count))
    dist = [np.inf] * vertex_count
    prev = [None] * vertex_count                 
    dist[source] = 0                       
     
    while Q:
        min_idx = np.argmin(dist)
        u = Q[min_idx]   
                                            
        Q.pop(min_idx)
        
        for i in range(vertex_count):
            if Graph[u, i] == None or u == i:
                continue
            alt = dist[u] + Graph[u, i]
            if dist[i] == None or alt < dist[i]:              
                dist[i] = alt
                prev[i] = u

    return dist, prev

def Dijkstra_heap(Graph, source):
    vertex_count = np.size(Graph, 0)
    dist = [np.inf] * vertex_count
    dist[source] = 0                       
    Q = DHeap.make_dheap(dist, np.arange(0, vertex_count))
    prev = [None] * vertex_count                 
     
    while Q:
        min_dist, min_dist_idx = Q.extractMin()   
        for i in range(vertex_count):
            if Graph[min_dist_idx, i] == None:
                continue
            alt = min_dist + Graph[min_dist_idx, i]
            if dist[i] == None or alt < dist[i]:              
                dist[i] = alt
                prev[i] = min_dist_idx

    return dist, prev

def generate_data(n, m, q, r):
    upper_indices = np.triu_indices(n, 1)
    upper_indices = np.array(list(zip(upper_indices[0], upper_indices[1])), dtype = 'i,i')
    upper_indices = np.random.choice(upper_indices, m, replace = False)

    graph = np.empty((n, n), dtype = object)
    
    for i in range(m):
       x, y = upper_indices[i]
       graph[x, y] = random.randint(q, r + 1)
       graph[y, x] = graph[x, y]
    
    for i in range(n):
        graph[i, i] = 0

    return graph 



def main():

    #graph = generate_data(10, 1, 10, 20)
    graph = np.array(
        [
            [0   , 7   , 9   , None, None, 14  ], 
            [7   , 0   , 10  , 15  , None, None],
            [9   , 10  , 0   , 11  , None, 2   ],
            [None, 15  , 11  , 0   , 6   , None],
            [None, None, None, 6   , 0   , 9   ],
            [14  , None, 2   , None, 9   , 0   ]
        ]
    )
    measurements_count = 100
    start = time.perf_counter()
    for i in range(measurements_count):
        Dijkstra(graph, 0)
    stop = time.perf_counter()
    elapsed_time = (stop - start) / measurements_count
    print(elapsed_time, 's')
    return

if __name__ == '__main__':
    main()