"""
Prior distributions on the radii of gyration of ensemble members
"""

from __future__ import print_function
import numpy

from csb.bio.utils import radius_of_gyration
from csb.statistics.pdf.parameterized import Parameter

from isd2 import ArrayParameter
from isd2.pdf.priors import AbstractPrior

class GyrationRadiusPrior(AbstractPrior):
    
    def __init__(self, name, rog, k_rog, n_structures):
        """
        Implements a prior distribution quadratically restraining 
        the radii of gyration of ensemble members to a certain value.

        :param name: a unique name for this object, e.g., 'rog_prior'
        :type name: str

        :param rog: target radius of gyration
        :type rog: float

        :param k_rog: force constant for harmonic potential
        :type k_rog: float

        :param n_structures: number of ensemble members
        :type n_structures: int
        """
        super(GyrationRadiusPrior, self).__init__(name)

        self.n_structures = n_structures
        
        self._register_variable('structures', differentiable=True)

        self._register('rog')
        self['rog'] = Parameter(rog, 'rog')
        self._register('k_rog')
        self['k_rog'] = Parameter(k_rog, 'k_rog')
        
        self.update_var_param_types(structures=ArrayParameter)
        self._set_original_variables()

    def _single_structure_log_prob(self, structure):
        """
        Evaluates log-probability for a single structure

        :param structure: coordinates of a single structure
        :type structure: numpy.ndarray of float; length: # beads * 3

        :returns: log-probability
        :rtype: float
        """
        X = structure.reshape(-1,3)
        rg = radius_of_gyration(X)

        return -0.5 * self['k_rog'].value * (self['rog'].value - rg) ** 2


    def _single_structure_gradient(self, structure):
        """
        Evaluates the negative log-probability gradient 
        for a single structure

        :param structure: coordinates of a single structure
        :type structure: numpy.ndarray of float; length: # beads * 3

        :returns: gradient vector
        :rtype: numpy.ndarray of float; length: # beads * 3
        """
        X = structure.reshape(-1,3)
        r_gyr = radius_of_gyration(X)
        k = self['k_rog'].value
        target_rog = self['rog'].value
        
        return -k * (target_rog - r_gyr) * (X - X.mean(0)).ravel() / r_gyr / len(X)
        
    def _evaluate_log_prob(self, structures):

        log_prob = self._single_structure_log_prob
        X = structures.reshape(self.n_structures, -1, 3)
		
        return numpy.sum(map(lambda x: log_prob(structure=x), X))

    def _evaluate_gradient(self, structures):

        grad = self._single_structure_gradient
        X = structures.reshape(self.n_structures, -1, 3)

        return numpy.concatenate(map(lambda x: grad(structure=x), X))

    def clone(self):

        copy = self.__class__(name=self.name,
                              rog=self['rog'].value,
                              k_rog=self['k_rog'].value,
                              n_structures=self.n_structures)

        copy.set_fixed_variables_from_pdf(self)

        return copy
        
    
if __name__ == '__main__':

    X = numpy.random.normal(size=(10,100,3), scale=5)

    P = GyrationRadiusPrior('asdfasd', rog=3.0, k_rog=10.0, n_structures=10)

    print(P.log_prob(structures=X.ravel()))
    
    from stuff import numgrad

    g = P.gradient(structures=X.ravel())
    ng = numgrad(X.ravel(), lambda x: -P.log_prob(structures=x))

    print(numpy.max(numpy.fabs(ng-g)))
