import numpy as np

from csb.statistics.pdf.parameterized import Parameter
from isd2.pdf.likelihoods import Likelihood as ISD2Likelihood

from .error_models import PoissonEM
from .likelihoods_c import calculate_gradient

class Likelihood(ISD2Likelihood):

    def __init__(self, name, fwm, em, lammda):

        super(Likelihood, self).__init__(name, fwm, em)

        self._register('lammda')
        self['lammda'] = Parameter(lammda, 'lammda')
        
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

        result = calculate_gradient(structures, smooth_steepness, 
                                    weights,
                                    fwm['contact_distances'].value,
                                    self.forward_model.data_points)
        
        return self['lammda'].value * result

    def clone(self):

        copy = self.__class__(self.name,
                              self.forward_model.clone(),
                              self.error_model.clone(),
                              self['lammda'].value)

        for v in copy.variables.difference(self.variables):
            copy._delete_variable(v)

        copy.set_fixed_variables_from_pdf(self)

        return copy

    def conditional_factory(self, **fixed_vars):

        fwm = self.forward_model.clone()
        fwm.fix_variables(**fwm._get_variables_intersection(fixed_vars))
        em = self.error_model.conditional_factory(**self.error_model._get_variables_intersection(fixed_vars))
        result = self.__class__(self.name, fwm, em,
                                self['lammda'].value)

        return result            
