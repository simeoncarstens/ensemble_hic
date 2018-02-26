import numpy, sys, os

from csb.statistics.pdf.parameterized import Parameter

from isd2 import ArrayParameter
from isd2.model.forwardmodels import AbstractForwardModel

from .forward_models_c import ensemble_contacts_evaluate
	

class EnsembleContactsFWM(AbstractForwardModel):

    def __init__(self, name, n_structures, contact_distances, data_points):

        super(EnsembleContactsFWM, self).__init__(name)

        self.data_points = data_points

        self.n_structures = n_structures

        self._register('contact_distances')
        self['contact_distances'] = ArrayParameter(contact_distances,
                                                   'contact_distance')

        self._register_variable('structures', differentiable=True)
        self._register_variable('smooth_steepness')
        self._register_variable('weights')
        self._register_variable('norm')
	    
        self.update_var_param_types(structures=ArrayParameter,
                                    smooth_steepness=Parameter, 
                                    weights=ArrayParameter,
                                    norm=Parameter)
        self._set_original_variables()

    def _evaluate(self, structures, smooth_steepness, weights, norm):

        X = structures.reshape(self.n_structures, -1, 3)

        if numpy.std(weights) == 0.0:
            weights = numpy.ones(self.n_structures) * norm

        return ensemble_contacts_evaluate(X,
                                          weights,
                                          self['contact_distances'].value, 
                                          smooth_steepness,
                                          data_points=numpy.array(self.data_points,
                                                                  dtype=int))
    
    def clone(self):

        copy = self.__class__(name=self.name, n_structures=self.n_structures, 
                              contact_distances=self['contact_distances'].value, 
                              data_points=self.data_points)

        for p in self.parameters:
            if not p in copy.parameters:
                copy._register(p)
                copy[p] = self[p].__class__(self[p].value, p)
                if p in copy.variables:
                    copy._delete_variable(p)                

        return copy
		
    def _evaluate_jacobi_matrix(self, structures, smooth_steepness, weights):

        raise NotImplementedError("Use fast likelihood gradients in " +
                                  "likelihoods_c.pyx instead!")    
