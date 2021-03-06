## cython: profile=True
# cython: wraparound=False
# cython: cdivision=True
# cython: boundscheck=False

import numpy
cimport numpy
cimport cython
from cython cimport view
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
                       double norm,
                       double [::1] cds, 
                       Py_ssize_t [:,::1] data_points):
    """
    A pure Cython implementation of the negative log-probability
    gradient of a likelihood with the usual forward model and a Poisson
    error model.
    """
    cdef double d, value, f, g
    cdef Py_ssize_t i,j,l,u,k,v, offset
    cdef Py_ssize_t n_datapoints = data_points.shape[0]
    cdef Py_ssize_t n_beads = structures.shape[1]
    cdef Py_ssize_t n_structures  = structures.shape[0]

    my_array2 = view.array(shape=(n_structures, n_datapoints),
                           itemsize=sizeof(double), format="d")
    cdef double [:,::1] distances = my_array2
    
    my_array3 = view.array(shape=(n_structures, n_datapoints),
                           itemsize=sizeof(double), format="d")
    cdef double [:,::1] sqrtdenoms = my_array3
    
    cdef double [::1] result = numpy.zeros(n_structures * n_beads * 3)
    cdef double [::1] md = numpy.zeros(n_datapoints)

    ## By changing this, you could use other error models. This is basically
    ## the derivative of the error model log-probability w.r.t the mock data
    em_derivative = poisson_derivative
    
    ensemble_contacts_evaluate_pureC(structures, norm, cds,
                                     alpha, data_points,
                                     distances, sqrtdenoms, md)
    
    for k in range(n_structures):
        offset = k * n_beads * 3
        for u in range(n_datapoints):
            i = data_points[u,0]
            j = data_points[u,1]
            d = distances[k,u]
            g = 1.0 + alpha * alpha * (cds[u] - d) * (cds[u] - d)
            f = 0.5 * em_derivative(md[u], data_points[u,2]) * norm * alpha / (g * sqrtdenoms[k,u] * d)
            for l in range(3):
                value = (structures[k,j,l] - structures[k,i,l]) * f
                result[offset + i * 3 + l] += value 
                result[offset + j * 3 + l] -= value

    return numpy.array(result)

