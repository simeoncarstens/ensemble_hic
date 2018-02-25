## cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
## cython: nonecheck = True

import numpy
cimport numpy
cimport cython

import numpy as np

from csb.statistics.pdf.parameterized import Parameter
from isd2.pdf.likelihoods import Likelihood as ISD2Likelihood

from .error_models import GaussianEM, LognormalEM, PoissonEM
from cpython.array cimport array, clone

cdef double [:] calculate_gradient(double [:,:,::1] structures,
                                    double smooth_steepness,
                                    double[::1] weights,
                                    double cutoff, double [::1] cds, 
                                    Py_ssize_t [:,::1] data_points,
                                    int em_indicator
                                    ):

    cdef double d, value, f, g, h
    cdef Py_ssize_t i,j,l,u,k,v
    cdef Py_ssize_t n_datapoints = len(data_points)
    cdef Py_ssize_t n_beads = len(structures[0])
    cdef Py_ssize_t n_structures  = len(structures)
    cdef double [::1] result = numpy.zeros(n_structures * n_beads * 3)
    cdef double [:,::1] distances = numpy.empty((n_structures, n_datapoints))
    cdef double [:,::1] sqrtdenoms = numpy.empty((n_structures, n_datapoints))
    cdef double [::1] md = numpy.zeros(n_datapoints)
    
    # cdef double [:] md = ensemble_contacts_evaluate(structures, weights, cds,
    #                                                 smooth_steepness, data_points,
    #                                                 cutoff, distances, sqrtdenoms)
    ensemble_contacts_evaluate(structures, weights, cds,
                               smooth_steepness, data_points,
                               cutoff, distances, sqrtdenoms, md)
    

    for u in range(n_datapoints):
        i = data_points[u,0]
        j = data_points[u,1]
        for k in range(n_structures):
            d = distances[k,u]
            
            g = 1.0 + smooth_steepness * smooth_steepness * (cds[u] - d) * (cds[u] - d)
            f = 0.5 * (1.0 - data_points[u,2] / md[u]) * weights[k] * smooth_steepness / (g * sqrtdenoms[k, u] * d)
            for l in range(3):
                value = (structures[k,j,l] - structures[k,i,l]) * f
                result[k * n_beads * 3 + i * 3 + l] += value 
                result[k * n_beads * 3 + j * 3 + l] -= value

    return result

cdef extern from "math.h" nogil:
    double sqrt(double)
        
cdef inline double ens_smooth(double x, double alpha):

    return (alpha * x / sqrt(1.0 + alpha * alpha * x * x) + 1.0) * 0.5

cdef inline double ens_dsmooth(double x, double alpha):

    return alpha / (1.0 + alpha * alpha * x * x) ** 1.5 * 0.5
    
cdef double [:] ensemble_contacts_evaluate(double [:,:,::1] structures, double [::1] weights,
                                           double [::1] contact_distances, 
                                           double alpha, Py_ssize_t [:,::1] data_points,
                                           double cutoff, 
                                           double [:,::1] distances, double [:,::1] sqrtdenoms, double [::1] res) nogil:
 
    cdef Py_ssize_t N = 20#len(structures)
    cdef Py_ssize_t n_data_points = 22366#len(data_points)
    cdef Py_ssize_t i, j, k, l, m
    cdef double d
    #cdef double [:] res = numpy.zeros(n_data_points)

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

    return res


class Likelihood(ISD2Likelihood):

    def __init__(self, name, fwm, em, lammda, gradient_cutoff=10000.0):

        super(Likelihood, self).__init__(name, fwm, em)

        self.gradient_cutoff = gradient_cutoff
        self._register('lammda')
        self['lammda'] = Parameter(lammda, 'lammda')
        
        if not True:
            dps = np.loadtxt('/usr/users/scarste/projects/ensemble_hic/data/rao2014/chr1_coarse.txt').astype(int)
            dps = dps[np.abs(dps[:,0] - dps[:,1]) > 1]
            dps = dps[dps[:,2] > 0]
            counts = dps[:,2] * 1.0
            counts /= 9.0
            dps[:,2] = counts.astype(int)
            self.dps = dps
        
    def gradient(self, **variables):

        self._complete_variables(variables)
        weights = self._get_weights(**variables)
        variables.update(weights=weights)
        result = self._evaluate_gradient(**variables)

        return result

    def log_prob(self, **variables):

        self._complete_variables(variables)
        weights = self._get_weights(**variables)
        variables.update(weights=weights)
        result = self._evaluate_log_prob(**variables)

        return self['lammda'].value * result
    
    def _get_weights(self, **variables):

        if 'weights' in self.variables and not 'norm' in self.variables:
            weights = variables['weights']
        elif 'norm' in self.variables and not 'weights' in self.variables:
            weights = np.ones(self.forward_model.n_structures) * variables['norm']
        elif 'weights' in self.variables and 'norm' in self.variables:
            raise('Can\'t have both norm and weights as variables!')
        elif not 'weights' in self.variables and not 'norm' in self.variables:
            weights = np.ones(self.forward_model.n_structures) * self['norm'].value
        else:
            raise('Something is wrong: can\'t decide how to set weights')

        return weights

    def _evaluate_gradient(self, **variables):
        
        structures = variables['structures'].reshape(self.forward_model.n_structures, -1, 3)
        smooth_steepness = variables['smooth_steepness']
        weights = variables['weights']

        fwm = self.forward_model

        if not True:
            dps = self.dps
        else:
            dps = fwm.data_points

        result = np.array(calculate_gradient(structures, smooth_steepness, 
                                             weights,
                                             self.gradient_cutoff,
                                             fwm['contact_distances'].value,
                                             dps,
                                             0
                                             ))
        
        return self['lammda'].value * result

    def _evaluate_log_prob(self, **variables):

        pass
        # fwm_variables, em_variables = self._split_variables(variables)
        # mock_data = ensemble_contacts_evaluate(fwm_variables['structures'].reshape(-1,213,3),
        #                                        fwm_variables['weights'], self.forward_model.contact_distances.value,
        #                                        self['smooth_steepness'].value,
        #                                        self.forward_model.data_points, 10.0, numpy.zeros((20, self.forward_model.data_points.shape[0])))
        
        # return self.error_model.log_prob(mock_data=mock_data)

    def clone(self):

        copy = self.__class__(self.name,
                              self.forward_model.clone(),
                              self.error_model.clone(),
                              self['lammda'].value,
                              self.gradient_cutoff)

        for v in copy.variables.difference(self.variables):
            copy._delete_variable(v)

        copy.set_fixed_variables_from_pdf(self)

        return copy

    def conditional_factory(self, **fixed_vars):

        fwm = self.forward_model.clone()
        fwm.fix_variables(**fwm._get_variables_intersection(fixed_vars))
        em = self.error_model.conditional_factory(**self.error_model._get_variables_intersection(fixed_vars))
        result = self.__class__(self.name, fwm, em,
                                self['lammda'].value, self.gradient_cutoff)

        return result            
