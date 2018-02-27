# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy
cimport numpy
cimport cython

cdef extern from "math.h" nogil:
    double sqrt(double)

cdef inline int ensemble_contacts_evaluate_pureC(double [:,:,::1] structures,
                                             double [::1] weights,	
                                             double [::1] contact_distances, 
                                             double alpha,
                                             Py_ssize_t [:,::1] data_points,
                                             double [:,::1] distances,
                                             double [:,::1] sqrtdenoms,
                                             double [::1] res) nogil:
 
    cdef Py_ssize_t N = structures.shape[0]
    cdef Py_ssize_t n_data_points = data_points.shape[0]
    cdef Py_ssize_t i, j, k, l, m
    cdef double d
 
    for m in range(n_data_points):
        i = data_points[m,0]
        j = data_points[m,1]
        for k in range(N):
            d = 0.0
            for l in range(3):
                d += (structures[k,i,l] - structures[k,j,l]) * (structures[k,i,l] - structures[k,j,l])
            distances[k,m] = sqrt(d)
            sqrtdenoms[k,m] = sqrt(1.0 + alpha * alpha * (contact_distances[m] - distances[k,m]) * (contact_distances[m] - distances[k,m]))
            
            res[m] += (alpha * (contact_distances[m] - distances[k,m]) / sqrtdenoms[k,m] + 1.0) * 0.5 * weights[k]

    return 1