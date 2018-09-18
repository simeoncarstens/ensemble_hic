# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy
cimport numpy
cimport cython

cdef extern from "math.h":
    double sqrt(double) nogil

def forcefield_energy(double [:,::1] s, double [:] radii, double [:] radii2,
                      double force_constant):

    """
    Cython implementation of a purely repulsive nonbonded potential
    in which distances below the sum of bead radii are penalized 
    quadratically
    """
    cdef Py_ssize_t i, j, l, k
    cdef double res = 0.0
    cdef int N = len(s)
    cdef double d
    
    for i in xrange(N):
        for j in xrange(i+1, N):
            d = (s[i,0] - s[j,0]) * (s[i,0] - s[j,0]) + (s[i,1] - s[j,1]) * (s[i,1] - s[j,1]) + (s[i,2] - s[j,2]) * (s[i,2] - s[j,2])
            if d < radii2[i] + radii2[j] + 2 * radii[i] * radii[j]:
                d = sqrt(d)
                res += (d - radii[i] - radii[j]) ** 4

    return 0.5 * force_constant * res

def forcefield_gradient(double [:,::1] s, double [:] radii, double [:] radii2,
                        double force_constant):

    """
    Cython implementation of the gradient of a purely repulsive nonbonded 
    potential in which distances below the sum of bead radii are penalized 
    quadratically
    """

    cdef Py_ssize_t i, j, l, k
    cdef double d, g, e
    cdef int N = len(s)
    cdef double [:,:] res = numpy.zeros((N, 3))

    for i in range(N):
        for j in range(i+1, N):
            d = 0.0
            for l in range(3):
                d += (s[i,l] - s[j,l]) * (s[i,l] - s[j,l])
            if d < radii2[i] + radii2[j] + 2 * radii[i] * radii[j]:
                d = sqrt(d)
                g = (radii[i] + radii[j] - d) * (radii[i] + radii[j] - d) * (radii[i] + radii[j] - d) / d
                for l in range(3):
                    res[i,l] -= (s[i,l] - s[j,l]) * g
                    res[j,l] += (s[i,l] - s[j,l]) * g
    
    return force_constant * 2.0 * numpy.array(res).ravel()
