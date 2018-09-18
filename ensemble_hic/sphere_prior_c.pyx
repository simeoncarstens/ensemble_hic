# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy
cimport numpy
cimport cython

cdef extern from "math.h":
    double sqrt(double)

def sphere_prior_gradient(double [:,:] structure, double [:] center,
                          double radius, double k, Py_ssize_t [:] beads,
                          double [:] bead_radii, double [:] bead_radii2):
    """
    Cython implementation of the gradient of a harmonic potential
    restraining beads to be located within a sphere
    """
    cdef Py_ssize_t i, j
    cdef double d, f
    cdef double radius2 = radius * radius
    cdef double [:,:] res = numpy.zeros((len(structure), 3))
    cdef Py_ssize_t n_beads = len(beads)
    
    for i in range(n_beads):
        d = 0.0
        for j in range(3):
            d += (structure[beads[i],j] - center[j]) * (structure[beads[i],j] - center[j])

        if d < radius2 + bead_radii2[i] - 2 * radius * bead_radii[i]:
            continue
        d = sqrt(d)

        f = (d + bead_radii[i] - radius) / d
        for j in range(3):
            res[i,j] += (structure[beads[i],j] - center[j]) * f

    return k * numpy.array(res).ravel()
