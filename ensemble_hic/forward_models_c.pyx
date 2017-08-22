# # cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy
cimport numpy
cimport cython

cdef extern from "math.h":
    double sqrt(double)
        
cdef inline double ens_smooth(double x, double alpha):

    return (alpha * x / sqrt(1.0 + alpha * alpha * x * x) + 1.0) * 0.5

cdef inline double ens_dsmooth(double x, double alpha):

    return alpha / (1.0 + alpha * alpha * x * x) ** 1.5 * 0.5
    
def ensemble_contacts_evaluate(double [:,:,:] structures, double [:] weights,
                               double [:] contact_distances, 
                               double alpha, Py_ssize_t [:,:] data_points,
                               double cutoff):

    cdef Py_ssize_t N = len(structures)
    cdef Py_ssize_t K = len(structures[0])
    cdef Py_ssize_t n_data_points = len(data_points)
    cdef Py_ssize_t i, j, k, l, m
    cdef double d
    cdef double [:] res = numpy.zeros(n_data_points)

    for m in range(n_data_points):
        i = data_points[m,1]
        j = data_points[m,2]
        for k in range(N):
            d = 0.0
            for l in range(3):
                d += (structures[k,i,l] - structures[k,j,l]) * (structures[k,i,l] - structures[k,j,l])
            res[m] += ens_smooth(contact_distances[m] - sqrt(d), alpha) * weights[k]

    return numpy.array(res)
