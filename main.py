import numpy as np
from numpy import random
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import math
import random
import scipy.sparse as sp

class DHeap:
    def __init__(self, arity = 9, alloc_size = 100, value = None, key = None):
        self.__arity = arity
        self.__heap = [None] * alloc_size
        self.__indices = {}
        self.__heap_size = 0
        if value:
            self.__value = value
        else:
            self.__value = lambda x: x
        
        if key:
            self.__key = key
        else:
            self.__key = lambda x: x

    def __len__(self):
        return self.__heap_size

    def siftDown(self, i):
        while self.__arity * i + 1 < self.__heap_size:
            min_child = - 1
            for j in range(self.__arity):
                child = self.__arity * i + j + 1
                if child >= self.__heap_size:
                    break
                if min_child == -1 or self.__value(self.__heap[min_child]) >= self.__value(self.__heap[child]):
                    min_child = child
            
            if self.__value(self.__heap[i]) > self.__value(self.__heap[min_child]):
                heap_i = self.__key(self.__heap[i])
                heap_min_child = self.__key(self.__heap[min_child])
                self.__indices[heap_i], self.__indices[heap_min_child] = self.__indices[heap_min_child], self.__indices[heap_i]
                self.__heap[i], self.__heap[min_child] = self.__heap[min_child], self.__heap[i]
                i = min_child
            else:
                break

    def siftUp(self, i):
        while True:
            parent_idx = int((i - 1) / self.__arity)
            child = self.__value(self.__heap[i])
            parent = self.__value(self.__heap[parent_idx])
            if child >= parent:
                break
            child_u = self.__key(self.__heap[i])
            parent_u = self.__key(self.__heap[parent_idx])
            self.__indices[child_u], self.__indices[parent_u] = self.__indices[parent_u], self.__indices[child_u]

            tmp = self.__heap[i]
            self.__heap[i] = self.__heap[parent_idx]
            self.__heap[int((i - 1) / self.__arity)] = tmp

            i = int((i - 1) / self.__arity)
    
    def extractMin(self):
        min = self.__heap[0]
        self.__indices.pop(self.__key(self.__heap[0]))
        self.__indices[self.__key(self.__heap[self.__heap_size - 1])] = 0
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
        self.__indices[self.__key(value)] = self.__heap_size - 1
        self.siftUp(self.__heap_size - 1)

    def __setitem__(self, value, obj):
        idx = self.__indices[value]
        old_priority = self.__value(self.__heap[idx])
        priority = self.__value(obj)
        self.__heap[idx] = obj
        if priority < old_priority:
            self.siftUp(idx)
        else:
            self.siftDown(idx)

    def Heapify(self, i):
        least = i
        for j in range(self.__arity):
            child = self.__arity * i + j + 1
            if child >= self.__heap_size:
                break
            if self.__value(self.__heap[child]) < self.__value(self.__heap[least]):
                least = child

        if least != i:
            heap_i = self.__key(self.__heap[i])
            heap_least = self.__key(self.__heap[least])
            self.__indices[heap_i], self.__indices[heap_least] = self.__indices[heap_least], self.__indices[heap_i]
            self.__heap[i], self.__heap[least] = self.__heap[least], self.__heap[i]
            self.Heapify(least)

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
    def make_dheap(values, arity = 9, value = None, key = None):
        dheap = DHeap(arity, alloc_size = 0, value=value, key = key)
        dheap.__heap_size = len(values)
        dheap.__heap = values
        for i, value_ in zip(range(len(values)), values):
            dheap.__indices[dheap.__key(value_)] = i
        dheap.__arity = arity
        dheap.Build_Heap()
        return dheap

def dijkstra(graph_csr : sp.csr_matrix, source):
    graph_csc = graph_csr.tocsc()
    vertex_count = graph_csr.get_shape()[0]
    Q = list(range(vertex_count))
    dist = [np.inf] * vertex_count                
    dist[source] = 0                       
    
    while Q:
        min_idx = np.argmin(dist)
        u = Q[min_idx]   
        Q.pop(min_idx)
        
        def update(graph):
            nnz = graph.indptr[u + 1] - graph.indptr[u]
            for k in range(nnz):
                i = graph.indptr[u] + k
                weight = graph.data[i]
                alt = dist[u] + weight
                if alt < dist[graph.indices[i]]:
                    dist[graph.indices[i]] = alt

        update(graph_csr)
        update(graph_csc)

    return dist

def dijkstra_heap(graph : sp.csr_matrix, source):
    graph_csc = graph.tocsc()
    vertex_count = np.size(graph, 0)
    dist = [np.inf] * vertex_count
    dist[source] = 0                       
    Q = DHeap.make_dheap(list(zip(range(vertex_count), dist)), value=lambda x: x[1], key=lambda x: x[0])
     
    while Q:
        u, _ = Q.extractMin()
        def update(graph):
            nnz = graph.indptr[u + 1] - graph.indptr[u]
            for k in range(nnz):
                i = graph.indptr[u] + k
                weight = graph.data[i]
                alt = dist[u] + weight
                if alt < dist[graph.indices[i]]:
                    dist[graph.indices[i]] = alt
                    Q[graph.indices[i]] = (graph.indices[i], alt)

        update(graph)
        update(graph_csc)

    return dist

def coor_to_idx(n, i, j):
    return i * (2 * n - i + 1) // 2 + j - i

def idx_to_coor(size, idx, offset = 0):
    n, k = size - offset, idx
    i = math.floor((-math.sqrt((2 * n + 1) * (2 * n + 1) - 8 * k) + 2 * n + 1) / 2)
    j = k + i - i * (2 * n - i + 1) // 2
    return i, j + offset

def generate_sparse_data(n, m, q, r):
    random_indices = [idx_to_coor(n, i, 1) for i in sorted(random.sample(range(n * (n - 1) // 2), m))]
    data = (r - q) * np.random.random_sample(m) + q
    indices = [coord[1] for coord in random_indices]
    indptr = [-1] * (n + 1)
    fst_nnz = -1
    fst_nnz_idx = 0
    for i, value in enumerate(random_indices):
        r, c = value
        indices[i] = c
        for _ in range(r - fst_nnz):
            indptr[fst_nnz_idx] = i
            fst_nnz = r
            fst_nnz_idx += 1
    for _ in range(n - fst_nnz):
        indptr[fst_nnz_idx] = m
        fst_nnz_idx += 1

    sgraph = sp.csr_matrix((data, indices, indptr), shape=(n, n), dtype=float)
    return sgraph

def generate_data(n, m, q, r):
    random_indices = [idx_to_coor(n, i, 1) for i in sorted(random.sample(range(n * (n - 1) // 2), m))]

    graph = np.empty((n, n), dtype = object)
    
    for i in range(m):
       x, y = random_indices[i]
       graph[x, y] = random.randint(q, r + 1)
       graph[y, x] = graph[x, y]
    
    for i in range(n):
        graph[i, i] = 0

    return graph  

def generate_debug_data():
    return np.array(
        [
            [0 , 7 , 9 , 0 , 0 , 14], 
            [7 , 0 , 10, 15, 0 , 0 ],
            [9 , 10, 0 , 11, 0 , 2 ],
            [0 , 15, 11, 0 , 6 , 0 ],
            [0 , 0 , 0 , 6 , 0 , 9 ],
            [14, 0 , 2 , 0 , 9 , 0 ]
        ]
    )

def generate_debug_sdata():
    return sp.csr_matrix(generate_debug_data())

def time_counter(func, measurements_count):

    start = time.perf_counter()
    for i in range(measurements_count):
        func()
    stop = time.perf_counter()

    return (stop - start) / measurements_count

def plots(vertices, time_heap, time_mark):
    
    fig = plt.figure()

    plt.ylabel('Elapsed time, seconds')
    plt.xlabel('Number of vertices')
    plt.plot(vertices, time_heap)

    plt.plot(vertices, time_mark)
    plt.legend(["d-heap", "marks"])

    plt.show()

def main():

    num_cores = 6
    def iter(i):
        j = min(1000 * i, int((i - 1) * i / 2))
        graph = generate_sparse_data(i, j, 1, 1e6)
        elapsed_time_mark = time_counter(lambda : dijkstra(graph, 0), 1)
        elapsed_time_heap = time_counter(lambda : dijkstra_heap(graph, 0), 1)
        return i, elapsed_time_heap, elapsed_time_mark
    processed_list = Parallel(n_jobs=num_cores)(delayed(iter)(i) for i in tqdm(range(1000, 10000 + 1, 1000)))
    vertices, time_heap, time_mark = zip(*processed_list)
    plots(vertices, time_heap, time_mark)
        
    return

if __name__ == '__main__':
    main()