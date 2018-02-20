## cython: profile=True
# cython: wraparound=False
# cython: cdivision=True
# cython: boundscheck=False

import numpy
cimport numpy
cimport cython

cdef extern from "math.h" nogil:
    double sqrt(double)

cdef extern from "math.h" nogil:
    double log(double)

cdef double ens_smooth(double x, double smooth_steepness):

    return (smooth_steepness * x / sqrt(1.0 + smooth_steepness * smooth_steepness * x * x) + 1.0) * 0.5    

cdef double lognormal_derivative(double md, double data):
    return log(md / data) / md

cdef double gaussian_derivative(double md, double data):
    return md - data

cdef double poisson_derivative(double md, double data):
    return 1.0 - data / md

def calculate_gradient(double [:,:,:] structures,
                       double smooth_steepness,
                       double[:] weights,
                       double cutoff, double [:] cds, 
                       Py_ssize_t [:,:] data_points,
                       int em_indicator
                       ):

    cdef double d, value, f, g
    cdef Py_ssize_t i,j,l,u,k,v
    cdef Py_ssize_t n_datapoints = len(data_points)
    cdef Py_ssize_t n_beads = len(structures[0])
    cdef Py_ssize_t n_structures  = len(structures)
    cdef double [:] result = numpy.zeros(n_structures * n_beads * 3)
    
    from .forward_models_c import ensemble_contacts_evaluate

    cdef double [:] md = ensemble_contacts_evaluate(structures, weights, cds,
                                                    smooth_steepness, data_points,
                                                    cutoff)

    if em_indicator == 0:
        em_derivative = gaussian_derivative
    elif em_indicator == 1:
        em_derivative = poisson_derivative
    elif em_indicator == 2:
        em_derivative = lognormal_derivative

    for u in range(n_datapoints):
        i = data_points[u,0]
        j = data_points[u,1]
        for k in range(n_structures):
            d = 0.0
            for l in range(3):
                d += (structures[k,i,l] - structures[k,j,l]) ** 2
            if d > (cds[u] * cds[u]) * cutoff:
                continue
            d = sqrt(d)
            g = 1.0 + smooth_steepness * smooth_steepness * (cds[u] - d) * (cds[u] - d)
            f = 0.5 * em_derivative(md[u], data_points[u,2]) * weights[k] * smooth_steepness / (g * sqrt(g) * d)
            # f = 0.5 * em_derivative(md[u], data_points[u,2]) * weights[k] * smooth_steepness / (1.0 + smooth_steepness * smooth_steepness * (cds[u] - d) * (cds[u] - d)) ** 1.5 / d
            for l in range(3):
                value = (structures[k,j,l] - structures[k,i,l]) * f
                result[k * n_beads * 3 + i * 3 + l] += value 
                result[k * n_beads * 3 + j * 3 + l] -= value

    return numpy.array(result)

