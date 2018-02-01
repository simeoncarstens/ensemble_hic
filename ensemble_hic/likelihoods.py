import numpy as np

from csb.statistics.pdf.parameterized import Parameter
from isd2.pdf.likelihoods import Likelihood as ISD2Likelihood

from .error_models import GaussianEM, LognormalEM, PoissonEM
from .likelihoods_c import calculate_gradient


GAUSSIAN_EM = 0
POISSON_EM = 1
LOGNORMAL_EM = 2

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
        if isinstance(self.error_model, GaussianEM):
            precision = variables['precision']
            em_indicator = GAUSSIAN_EM
        elif isinstance(self.error_model, LognormalEM):
            precision = variables['precision']
            em_indicator = LOGNORMAL_EM
        elif isinstance(self.error_model, PoissonEM):
            precision = 1.0
            em_indicator = POISSON_EM

        fwm = self.forward_model

        if not True:
            dps = self.dps
        else:
            dps = fwm.data_points

        result = precision * calculate_gradient(structures, smooth_steepness, 
                                                weights,
                                                self.gradient_cutoff,
                                                fwm['contact_distances'].value,
                                                dps,
                                                em_indicator
                                                )
        
        return self['lammda'].value * result

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
