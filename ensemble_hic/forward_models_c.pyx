# # cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy
cimport numpy
cimport cython
from .evaluate_FWM_pureC cimport ensemble_contacts_evaluate_pureC

cdef extern from "math.h" nogil:
    double sqrt(double)
        
def ensemble_contacts_evaluate(double [:,:,::1] structures,
                               double norm,
                               double [::1] contact_distances, 
                               double alpha,
                               Py_ssize_t [:,::1] data_points):
    """
    Cython implementation of the only implemented forward model
    """
    cdef double [:,::1] distances = numpy.empty((structures.shape[0],
                                                 data_points.shape[0]))
    cdef double [:,::1] sqrtdenoms = numpy.empty((structures.shape[0],
                                                  data_points.shape[0]))
    cdef double [::1] res = numpy.zeros(data_points.shape[0])

    ensemble_contacts_evaluate_pureC(structures, norm, contact_distances,
                                     alpha, data_points,
                                     distances, sqrtdenoms, res)
    
    return numpy.array(res)


