import numpy as np

from abc import abstractmethod

from csb.statistics.pdf.parameterized import Parameter
from csb.numeric import log as csb_log

from isd2.pdf.priors import AbstractPrior
from isd2 import ArrayParameter

from .forcefield_c import forcefield_energy, forcefield_gradient


class AbstractNonbondedPrior(AbstractPrior):

    def __init__(self, name, bead_radii, force_constant, n_structures):

        super(AbstractNonbondedPrior, self).__init__(name)

        self.n_structures = n_structures
        self.bead_radii = bead_radii
        self.bead_radii2 = bead_radii ** 2
        
        self._register_variable('structures', differentiable=True)
        self._register('nonbonded_k')
        self['nonbonded_k'] = Parameter(force_constant, 'nonbonded_k')
        self.update_var_param_types(structures=ArrayParameter)
        self._set_original_variables()

    @abstractmethod
    def _register_ensemble_parameters(self, **parameters):
        pass

    def _forcefield_gradient(self, structure):

        grad = forcefield_gradient(structure, self.bead_radii,
                                   self.bead_radii2,
                                   self['nonbonded_k'].value)
        
        return grad

    def _forcefield_energy(self, structure):

        E = forcefield_energy(structure, self.bead_radii,
                              self.bead_radii2,
                              self['nonbonded_k'].value)

        return E

    @abstractmethod
    def _log_ensemble_gradient(self, E):
        pass
    
    @abstractmethod
    def _log_ensemble(self, E):
        pass    
        
    def _evaluate_log_prob(self, structures):

        ff_E = self._forcefield_energy
        log_ens = self._log_ensemble
        X = structures.reshape(self.n_structures, -1, 3)

        return -np.sum(map(lambda x: log_ens(ff_E(structure=x)), X))

    def _evaluate_gradient(self, structures):

        X = structures.reshape(self.n_structures, -1, 3)

        res = np.zeros((X.shape[0], X.shape[1] * 3))
        for i, x in enumerate(X):
            evald_ff_E = self._forcefield_energy(structure=x)
            evald_ff_grad = self._forcefield_gradient(structure=x)
            res[i] = self._log_ensemble_gradient(evald_ff_E) * evald_ff_grad

        return -res.ravel()

    @abstractmethod
    def clone(self):
        pass


class BoltzmannNonbondedPrior(AbstractNonbondedPrior):

    def __init__(self, name, bead_radii, force_constant, n_structures, beta):

        super(BoltzmannNonbondedPrior, self).__init__(name, bead_radii,
                                                      force_constant,
                                                      n_structures)
        self._register_ensemble_parameters(beta)

    def _register_ensemble_parameters(self, beta):

        self._register('beta')
        self['beta'] = Parameter(beta, 'beta')

    def _log_ensemble(self, E):

        return -self['beta'].value * E

    def _log_ensemble_gradient(self, E):

        return -self['beta'].value

    def clone(self):

        copy = self.__class__(self.name,
                              self.bead_radii,
                              self['nonbonded_k'].value, 
                              self.n_structures,
                              self['beta'].value)
        copy.set_fixed_variables_from_pdf(self)

        return copy


class TsallisNonbondedPrior(AbstractNonbondedPrior):

    def __init__(self, name, bead_radii, force_constant, n_structures, q):

        super(TsallisNonbondedPrior, self).__init__(name, bead_radii,
                                                      force_constant,
                                                      n_structures)
        self._register_ensemble_parameters(q)

    def _register_ensemble_parameters(self, q):

        self._register('q')
        self['q'] = Parameter(q, 'q')

    def _log_ensemble(self, E):

        q = self['q'].value

        if q == 1.0:
            return -E
        else:
            return -q * csb_log(1.0 + (q - 1.0) * E) / (q - 1.0)
    
    def _log_ensemble_gradient(self, E):

        q = self['q'].value

        if q == 1.0:
            return -E
        else:
            return -q / (1.0 + (q - 1.0) * E)
 
    def clone(self):

        copy = self.__class__(self.name,
                              self.bead_radii,
                              self['nonbonded_k'].value, 
                              self.n_structures,
                              self['q'].value)
        copy.set_fixed_variables_from_pdf(self)

        return copy


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from stuff import numgrad
    
    n_structures = 3
    n_beads = 70
    bead_radii = np.ones(n_beads) * 0.5
    X = np.random.uniform(low=-5, high=5, size=(n_structures, n_beads, 3))
    X = X.ravel()

    if not True:
        ## test Boltzmann ensemble
        P = BoltzmannNonbondedPrior('bla', bead_radii, 3.123, n_structures, 2.11)
    else:
        ## test Tsallis ensemble
        P = TsallisNonbondedPrior('bla', bead_radii, 3.123, n_structures, 1.03)
    g = P.gradient(structures=X)
    ng = numgrad(X, lambda x: -P.log_prob(structures=x))
    print np.max(np.fabs(g-ng))
    plt.scatter(g, ng)
    plt.show()
