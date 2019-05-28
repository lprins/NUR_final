import numpy as np

def swap(A, a, b):
    temp = A[a]
    A[a] = A[b]
    A[b] = temp

def selection_sort(A):
    N = len(A)
    for i in range(N - 1):
        min_el = i
        for j in range(i+1, N):
            if A[j] < A[min_el]:
                min_el = j
        swap(A, i, min_el)
def selection_argsort(A):
    N = len(A)
    indices = np.arange(N, dtype=int)
    for i in range(N - 1):
        min_el = i
        for j in range(i+1, N):
            if A[indices[j]] < A[indices[min_el]]:
                min_el = j
        swap(indices, i, min_el)
    return indices

def _partition(A):
    N = len(A)
    pivot_el = A[N >> 1]
    i = 0
    while(True):
        while(A[i] < pivot_el):
            i += 1
        j = N-1
        while(A[j] > pivot_el):
            j -= 1
        if i == j:
            return i
        swap(A, i, j)

def quicksort(A):
    # DO NOT USE
    # Some (unknown) error makes it slow
    N = len(A)
    if N < 10:
        selection_sort(A)
        return
    #Use median of three
    middle = N >> 1
    if A[middle] < A[0]:
        swap(A, 0, middle)
    if A[-1] < A[middle]:
        swap(A, -1, middle)
    if A[middle] < A[0]:
        swap(A, 0, middle)
    # Element in middle is now median of first, middle last elements
    # This is pivot
    p = _partition(A)
    quicksort(A[:p])
    quicksort(A[p+1:])

def _merge(in1, in2, out):
    N1 = len(in1)
    N2 = len(in2)
    N = N1 + N2
    j1 = 0
    j2 = 0
    for i in range(N):
        try:
            if in1[j1] < in2[j2]:
                out[i] = in1[j1]
                j1 += 1
            else:
                out[i] = in2[j2]
                j2 += 1
        except IndexError:
            # We are at the end of one of the lists
            # Now just append the other list onto out
            if j1 == N1:
                out[i:] = in2[j2:]
            else:
                out[i:] = in1[j1:]
            return

def mergesort(A):
    N = len(A)
    # Lists of size 1 (or 0) are trivially sorted
    # However for small lists selection sort is more efficient
    if N < 10:
        selection_sort(A)
        return
    middle = N >> 1
    mergesort(A[:middle])
    mergesort(A[middle:])
    _merge(np.copy(A[:middle]), np.copy(A[middle:]), A)

def _heap_parent(n):
    return (n - 1) // 2
def _heap_left_child(n):
    return 2 * n + 1

def _construct_heap(A):
    N = len(A)
    for start in range(_heap_parent(N-1), -1, -1):
        _repair_heap(A, start, N)

def _repair_heap(A, start, end):
    # Repair the heap in A between start and end with incorrect top node
    pos = start
    to_swap = pos
    left_child = _heap_left_child(pos)
    while left_child < end:
        if A[to_swap] < A[left_child]:
            to_swap = left_child
        if left_child + 1 < end and A[to_swap] < A[left_child + 1]:
            to_swap = left_child + 1
        if to_swap == pos:
            return
        swap(A, pos, to_swap)
        pos = to_swap
        left_child = _heap_left_child(pos)

def heapsort(A):
    N = len(A)
    _construct_heap(A)
    for heap_end in range(N-1, 0, -1):
        swap(A, 0, heap_end)
        _repair_heap(A, 0, heap_end)
