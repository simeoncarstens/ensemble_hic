# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy
cimport numpy
cimport cython

cdef extern from "math.h":
    double sqrt(double) nogil

def backbone_prior_gradient(double [:] structure,
                            double [:] lower_limits,
                            double [:] upper_limits,
                            double force_constant):
    """
    Cython implementation of the gradient of a potential
    penalizing too low / high distances between consecutive
    beads quadratically
    """
    cdef Py_ssize_t i, j
    cdef int N = len(structure) / 3
    cdef double d, e
    cdef double [:,:] s = numpy.reshape(structure, (N,3))
    cdef double [:,:] res = numpy.zeros((N, 3))

    for i in range(1, N):
        d = 0.0
        for j in range(3):
            d += (s[i, j] - s[i-1,j]) * (s[i, j] - s[i-1,j])

        if d > upper_limits[i-1] * upper_limits[i-1]:
            d = sqrt(d)
            for j in range(3):
                e = (d - upper_limits[i-1]) * (s[i,j] - s[i-1,j]) / d
                res[i,j] += e
                res[i-1,j] -= e
        elif d < lower_limits[i-1] * lower_limits[i-1]:
            d = sqrt(d)
            for j in range(3):
                e = (lower_limits[i-1] - d) * (s[i,j] - s[i-1,j]) / d
                res[i,j] -= e
                res[i-1,j] += e

    return force_constant * numpy.array(res).flatten()

