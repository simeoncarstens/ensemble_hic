# # cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy
cimport cython

cdef extern from "math.h" nogil:
    double sqrt(double)


cdef double contact_function(double d, double a, double cutoff):

    return 0.5 * a * (cutoff - d) / sqrt(1 + a * a * (cutoff - d) * (cutoff - d)) + 0.5


cdef double distance(double[:] x, double[:] y):

    return sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]))


cdef double[:,:] cgen_ss(double[:,:] x, double a, double cutoff, Py_ssize_t offset):

    cdef Py_ssize_t i, j
    cdef double d
    cdef double[:,:] result = np.zeros((308, 308))

    for i in range(308):
        for j in range(i+1, 308):
            if j - i > offset:
                d = distance(x[i], x[j])
                result[i,j] = contact_function(d, a, cutoff)
    
    return result


cdef double[:,:] cgen_pop(double[:,:,:] X, double a, double cutoff, Py_ssize_t offset):

    cdef Py_ssize_t i, j, k, N
    cdef double d
    cdef double[:,:] result = np.zeros((308, 308))
    N = len(X)
    
    for k in range(N):
        for i in range(308):
            for j in range(i+1, 308):
                if j - i > offset:
                    d = distance(X[k,i], X[k,j])
                    result[i,j] += contact_function(d, a, cutoff) / N
    
    return result



cdef double calculate_area(double[:,:] c, double a, double cutoff, Py_ssize_t offset):

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t n_beads = len(c)
    cdef double[:] counts = np.zeros(len(c))
    inds = np.arange(n_beads)
    areas = inds ** 2 + (n_beads - inds) ** 2
    
    for i in range(n_beads):
        for j in range(i):
            for k in range(j+1, i):
                counts[i] += c[j,k]

        for j in range(i, n_beads):
            for k in range(j+1, n_beads):
                counts[i] += c[j,k]
                
        
    return np.argmax(np.array(counts).astype('d') / np.array(areas))


def find_TADs(double[:,:,:] X, double a, double cutoff, Py_ssize_t offset):

    cdef double[:] result = np.zeros(len(X))
    cdef Py_ssize_t i
    cdef double[:,:] c

    for i in range(len(result)):
        c = cgen_ss(X[i], a, cutoff, offset)
        result[i] = calculate_area(c, a, cutoff, offset)
        
    return np.array(result)


def find_TADs_pop(double[:,:,:,:] Y, double a, double cutoff, Py_ssize_t offset):

    cdef double[:] result = np.zeros(len(Y))
    cdef Py_ssize_t i
    cdef double[:,:] c

    for i in range(len(result)):
        c = cgen_pop(Y[i], a, cutoff, offset)
        result[i] = calculate_area(c, a, cutoff, offset)

    return np.array(result)
