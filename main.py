import numpy as np
from numpy import random
import time

from numpy.lib.function_base import insert

class DHeap:
    def __init__(self, arity = 9, alloc_size = 100, key_selector = None, unique_selector = None):
        self.__arity = arity
        self.__heap = [None] * alloc_size #Первоначальный размер памяти, выделенной под кучу
        self.__indices = {}
        self.__heap_size = 0
        if key_selector:
            self.__select_key = key_selector
        else:
            self.__select_key = lambda x: x
        
        if unique_selector:
            self.__select_unique = unique_selector
        else:
            self.__select_unique = lambda x: x

    def __len__(self):
        return self.__heap_size

    def siftDown(self, i):
        while self.__arity * i + 1 < self.__heap_size:
            min_child = - 1
            for j in range(self.__arity):
                child = self.__arity * i + j + 1
                if child >= self.__heap_size:
                    break
                if min_child == -1 or self.__select_key(self.__heap[min_child]) >= self.__select_key(self.__heap[child]):
                    min_child, child = child, min_child
            
            heap_i = self.__select_unique(self.__heap[i])
            heap_min_child = self.__select_unique(self.__heap[min_child])
            if heap_i > heap_min_child:
                self.__indices[heap_i], self.__indices[heap_min_child] = self.__indices[heap_min_child], self.__indices[heap_i]
                self.__heap[i], self.__heap[min_child] = self.__heap[min_child], self.__heap[i]
                i = min_child
            else:
                break
                
        #Найти сына, который меньше текущего элемента
        #Если найдено, меняем местами текущий эл-т и сына
        #Выполнить siftDown для этого сына


    def siftUp(self, i):
        heap_i = self.__select_unique(self.__heap[i])
        parent = self.__select_unique(self.__heap[int((i - 1) / self.__arity)])
        while heap_i < parent:
            self.__indices[heap_i], self.__indices[parent] = self.__indices[parent], self.__indices[heap_i]
            self.__heap[i], self.__heap[int((i - 1) / self.__arity)] = self.__heap[int((i - 1) / self.__arity)], self.__heap[i]
            i = int((i - 1) / self.__arity)
    
    def extractMin(self):
        min = self.__heap[0]
        self.__indices.pop(self.__select_unique(self.__heap[0]))
        self.__indices[self.__select_unique(self.__heap[self.__heap_size - 1])] = 0
        self.__heap[0] = self.__heap[self.__heap_size - 1]
        self.__heap_size = self.__heap_size - 1
        self.siftDown(0)
        return min

    def __reallocate_heap(self, new_heap_size):
        if new_heap_size >= np.size(self.__heap):
            heap = [None] * (len(self.__heap) * 2)
            heap[:len(self.__heap)] = self.__heap
            self.__heap = heap
    
    def insert(self, value):
        self.__reallocate_heap(self.__heap_size + 1)
        self.__heap_size = self.__heap_size + 1
        self.__heap[self.__heap_size - 1] = value
        self.__indices[self.__select_unique(value)] = self.__heap_size - 1
        self.siftUp(self.__heap_size - 1)

    def __setitem__(self, value, priority):
        if value in self.__indices:
            idx = self.__indices[value]
            self.__heap[idx] = priority
            self.Heapify(idx)
        else:
            self.insert(priority)

    def Heapify(self, i):
        least = i
        for j in range(self.__arity):
            child = self.__arity * i + j + 1
            #heap_size - количество элементов в куче
            if child >= self.__heap_size:
                break
            if self.__select_key(self.__heap[child]) < self.__select_key(self.__heap[least]):
                least = child

        if least != i:
            heap_i = self.__select_unique(self.__heap[i])
            heap_least = self.__select_unique(self.__heap[least])
            self.__indices[heap_i], self.__indices[heap_least] = self.__indices[heap_least], self.__indices[heap_i]
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
            a.__indices[b.__select_unique(b.__heap[i])] = a.__heap_size - 1
        a.Build_Heap()

    @staticmethod
    def make_dheap(values, arity = 9, key_selector = None, unique_selector = None):
        dheap = DHeap(arity, alloc_size = 0, key_selector=key_selector, unique_selector = unique_selector)
        dheap.__heap_size = len(values)
        dheap.__heap = values
        for i, value in zip(range(len(values)), values):
            dheap.__indices[dheap.__select_unique(value)] = i
        dheap.__arity = arity
        dheap.Build_Heap()
        return dheap


def Dijkstra(Graph, source):
    vertex_count = np.size(Graph, 0)
    Q = list(range(vertex_count))
    dist = [np.inf] * vertex_count
    prev = [None] * vertex_count                 
    dist[source] = 0                       
    
    start = time.perf_counter()
    while Q:
        min_idx = np.argmin(dist)
        u = Q[min_idx]   
                                            
        Q.pop(min_idx)
        
        for i in range(vertex_count):
            if Graph[u, i] == None or u == i:
                continue
            alt = dist[u] + Graph[u, i]
            if alt < dist[i]:              
                dist[i] = alt
                prev[i] = u
    stop = time.perf_counter()
    elapsed_time = (stop - start)
    print('dijkstra elapsed_time: {} s'.format(elapsed_time))

    return dist, prev

def Dijkstra_heap(Graph, source):
    vertex_count = np.size(Graph, 0)
    dist = [np.inf] * vertex_count
    dist[source] = 0                       
    Q = DHeap.make_dheap(list(zip(range(vertex_count), dist)), key_selector=lambda x: x[1], unique_selector=lambda x: x[0])
    prev = [None] * vertex_count
     
    start = time.perf_counter()
    while Q:
        vertex_idx, min_dist = Q.extractMin()
        for i in range(vertex_count):
            if Graph[vertex_idx, i] == None:
                continue
            alt = min_dist + Graph[vertex_idx, i]
            if alt < dist[i]:              
                dist[i] = alt
                prev[i] = vertex_idx
                Q[i] = (i, alt)

    stop = time.perf_counter()
    elapsed_time = (stop - start)
    print('dijkstra on heap elapsed_time: {} s'.format(elapsed_time))
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
    graph = generate_data(10000, 5000, 1, 2000)
    #graph = np.array(
    #    [
    #        [0   , 7   , 9   , None, None, 14  ], 
    #        [7   , 0   , 10  , 15  , None, None],
    #        [9   , 10  , 0   , 11  , None, 2   ],
    #        [None, 15  , 11  , 0   , 6   , None],
    #        [None, None, None, 6   , 0   , 9   ],
    #        [14  , None, 2   , None, 9   , 0   ]
    #    ]
    #)
    measurements_count = 1
    
    _, _ = Dijkstra(graph, 0)
    _, _ = Dijkstra_heap(graph, 0)
    
    return

if __name__ == '__main__':
    main()