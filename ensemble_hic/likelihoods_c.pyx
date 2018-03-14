## cython: profile=True
# cython: wraparound=False
# cython: cdivision=True
# cython: boundscheck=False

import numpy
cimport numpy
cimport cython
from .evaluate_FWM_pureC cimport ensemble_contacts_evaluate_pureC

cdef extern from "math.h" nogil:
    double sqrt(double)

cdef extern from "math.h" nogil:
    double log(double)

cdef double lognormal_derivative(double md, double data):
    return log(md / data) / md

cdef double gaussian_derivative(double md, double data):
    return md - data

cdef double poisson_derivative(double md, double data):
    return 1.0 - data / md

def calculate_gradient(double [:,:,::1] structures,
                       double alpha,
                       double[::1] weights,
                       double [::1] cds, 
                       Py_ssize_t [:,::1] data_points):

    cdef double d, value, f, g
    cdef Py_ssize_t i,j,l,u,k,v
    cdef Py_ssize_t n_datapoints = data_points.shape[0]
    cdef Py_ssize_t n_beads = structures.shape[1]
    cdef Py_ssize_t n_structures  = structures.shape[0]
    cdef double [::1] result = numpy.zeros(n_structures * n_beads * 3)
    cdef double [:,::1] distances = numpy.empty((n_structures, n_datapoints))
    cdef double [:,::1] sqrtdenoms = numpy.empty((n_structures, n_datapoints))
    cdef double [::1] md = numpy.zeros(n_datapoints)
    em_derivative = poisson_derivative
    
    ensemble_contacts_evaluate_pureC(structures, weights, cds,
                                     alpha, data_points,
                                     distances, sqrtdenoms, md)
    
    for u in range(n_datapoints):
        i = data_points[u,0]
        j = data_points[u,1]
        for k in range(n_structures):
            d = distances[k,u]
            g = 1.0 + alpha * alpha * (cds[u] - d) * (cds[u] - d)
            f = 0.5 * em_derivative(md[u], data_points[u,2]) * weights[k] * alpha / (g * sqrtdenoms[k,u] * d)

            for l in range(3):
                value = (structures[k,j,l] - structures[k,i,l]) * f
                result[k * n_beads * 3 + i * 3 + l] += value 
                result[k * n_beads * 3 + j * 3 + l] -= value

    return numpy.array(result)

