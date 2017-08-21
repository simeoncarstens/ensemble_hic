import numpy

from csb.statistics.pdf.parameterized import Parameter

from isd2.pdf.priors import AbstractPrior
from isd2 import ArrayParameter

from .nonbonded_prior_c import nonbonded_prior_log_prob, nonbonded_prior_gradient


class NonbondedPrior(AbstractPrior):

    def __init__(self, name, bead_radii, force_constant, beta, n_structures):

        super(NonbondedPrior, self).__init__(name)

        self.n_structures = n_structures
        self.bead_radii = bead_radii
        self.bead_radii2 = bead_radii ** 2
        
        self._register_variable('structures', differentiable=True)
        self._register('beta')
        self['beta'] = Parameter(beta, 'beta')
        self._register('nonbonded_k')
        self['nonbonded_k'] = Parameter(force_constant, 'nonbonded_k')
        self.update_var_param_types(structures=ArrayParameter)
        self._set_original_variables()

    def _single_structure_gradient(self, structure):

        grad = nonbonded_prior_gradient(structure, self.bead_radii, self.bead_radii2,
                                        self['nonbonded_k'].value)
        
        return self['beta'].value * grad

    def _single_structure_log_prob(self, structure):

        log_prob = nonbonded_prior_log_prob(structure, self.bead_radii,
                                            self.bead_radii2,
                                            self['nonbonded_k'].value)

        return self['beta'].value * log_prob
        
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
                              self.bead_radii,
                              self['nonbonded_k'].value, 
                              self['beta'].value,
                              self.n_structures)
        copy.set_fixed_variables_from_pdf(self)

        return copy
