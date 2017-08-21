import numpy, os, sys

from csb.statistics.pdf.parameterized import Parameter

from isd2.pdf.priors import AbstractPrior

from .backbone_prior_c import backbone_prior_gradient


class BackbonePrior(AbstractPrior):

    def __init__(self, name, lower_limits, upper_limits, k_bb, n_structures):

        from isd2 import ArrayParameter
        from csb.statistics.pdf.parameterized import Parameter

        super(BackbonePrior, self).__init__(name)

        self.n_structures = n_structures
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self._register('k_bb')
        self['k_bb'] = Parameter(k_bb, 'k_bb')

        self._register_variable('structures', differentiable=True)
        self.update_var_param_types(structures=ArrayParameter)
        self._set_original_variables()

    def _single_structure_log_prob(self, structure):

        x = structure.reshape(-1, 3)
        k_bb = self['k_bb'].value
        ll = self.lower_limits
        ul = self.upper_limits

        d = numpy.sqrt(numpy.sum((x[1:] - x[:-1]) ** 2, 1))
        
        u_viols = d > ul
        l_viols = d < ll
        delta = ul - ll

        return -0.5 * k_bb * (  numpy.sum((d[u_viols] - ul[u_viols]) ** 2 ) \
                              + numpy.sum((ll[l_viols] - d[l_viols]) ** 2))
    
    def _single_structure_gradient(self, structure):
        
        return backbone_prior_gradient(structure.ravel(), 
                                       self.lower_limits, 
                                       self.upper_limits, 
                                       self['k_bb'].value)
        
    def _evaluate_log_prob(self, structures):

        log_prob = self._single_structure_log_prob
        X = structures.reshape(self.n_structures, -1, 3)
		
        return numpy.sum(map(lambda x: log_prob(structure=x), X))

    def _evaluate_gradient(self, structures):

        grad = self._single_structure_gradient
        X = structures.reshape(self.n_structures, -1, 3)

        return numpy.concatenate(map(lambda x: grad(structure=x), X))
		
    def clone(self):

        copy = self.__class__(self.name,
                              self.lower_limits,
                              self.upper_limits,
                              self['k_bb'].value, 
                              self.n_structures)

        copy.fix_variables(**{p: self[p].value for p in self.parameters
                              if not p in copy.parameters})

        return copy


