"""
Prior distributions for non-bonded interactions between beads
"""
import numpy as np

from abc import abstractmethod

from csb.statistics.pdf.parameterized import Parameter
from csb.numeric import log as csb_log

from isd2.pdf.priors import AbstractPrior
from isd2 import ArrayParameter


class AbstractNonbondedPrior2(AbstractPrior):

    def __init__(self, name, forcefield, n_structures):
        """
        General class implementing non-bonded priors.
        Subclasses implement different ensembles like the
        Boltzmann or the Tsallis ensemble.

        :param name: a unique name for this object, usually
                     'nonbonded_prior'
        :type name: str

        :param forcefield: a force field object describing the
                           physical interaction between beads
        :type forcefield: subclass of :ref:`.AbstractForceField`

        :param n_structures: number of ensemble members
        :type n_structures: int
        """
        super(AbstractNonbondedPrior2, self).__init__(name)

        self.n_structures = n_structures
        self.forcefield = forcefield
        
        self._register_variable('structures', differentiable=True)
        self.update_var_param_types(structures=ArrayParameter)
        self._set_original_variables()

    @abstractmethod
    def _register_ensemble_parameters(self, **parameters):
        """
        Register parameters of the statistical ensemble, for example
        the inverse temperature in case of a Boltzmann ensemble
        """
        pass

    def _forcefield_gradient(self, structure):
        """
        Evaluates the gradient of the force field

        :param structure: coordinates of structure ensemble
        :type structure: numpy.ndarray of floats; length: # beads * 3

        :returns: gradient vector
        :rtype: numpy.ndarray of floats; length: # beads * 3
        """

        return self.forcefield.gradient(structure)

    def _forcefield_energy(self, structure):
        """
        Evaluates the energy of the force field

        :param structure: coordinates of structure ensemble
        :type structure: numpy.ndarray of floats; length: # beads * 3

        :returns: force field energy
        :rtype: float
        """

        return self.forcefield.energy(structure)

    @abstractmethod
    def _log_ensemble_gradient(self, E):
        """
        Derivative of the statistical ensemble w.r.t. the system energy.

        Should be called log_ensemble_derivative or sth. like that.

        :param E: system energy calculated by a force field object
        :type E: float

        :returns: derivative w.r.t. the energy
        :rtype: float
        """
        pass
    
    @abstractmethod
    def _log_ensemble(self, E):
        """
        The logarithm of the statistical ensemble, for example,
        -beta * E in case of a Boltzmann ensemble

        :param E: system energy calculated by a force field object
        :type E: float
        """
        pass    
        
    def _evaluate_log_prob(self, structures):

        ff_E = self._forcefield_energy
        log_ens = self._log_ensemble
        X = structures.reshape(self.n_structures, -1, 3)

        return np.sum(map(lambda x: log_ens(ff_E(structure=x)), X))

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


class BoltzmannNonbondedPrior2(AbstractNonbondedPrior2):

    def __init__(self, name, forcefield, n_structures, beta):
        """
        Class implementing a non-bonded prior as a Boltzmann
        ensemble.

        :param name: a unique name for this object, usually
                     'nonbonded_prior'
        :type name: str

        :param forcefield: a force field object describing the
                           physical interaction between beads
        :type forcefield: subclass of :ref:`.AbstractForceField`

        :param n_structures: number of ensemble members
        :type n_structures: int

        :param beta: inverse temperature
        :type beta: float
        """
        super(BoltzmannNonbondedPrior2, self).__init__(name, forcefield,
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
                              self.forcefield, 
                              self.n_structures,
                              self['beta'].value)
        copy.set_fixed_variables_from_pdf(self)

        return copy



class TsallisNonbondedPrior(AbstractNonbondedPrior):

    def __init__(self, name, bead_radii, force_constant, n_structures, q):
        """
        Class implementing a non-bonded prior as a Tsallis
        ensemble.

        :param name: a unique name for this object, usually
                     'nonbonded_prior'
        :type name: str

        :param forcefield: a force field object describing the
                           physical interaction between beads
        :type forcefield: subclass of :ref:`.AbstractForceField`

        :param n_structures: number of ensemble members
        :type n_structures: int

        :param beta: inverse temperature
        :type beta: float
        """
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
            return -csb_log(1.0 + (q - 1.0) * E) / (q - 1.0)
    
    def _log_ensemble_gradient(self, E):

        q = self['q'].value

        if q == 1.0:
            return -E
        else:
            return -(1.0 + (q - 1.0) * E)
 
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
    
    n_structures = 1
    n_beads = 70
    bead_radii = np.ones(n_beads) * 0.5
    X = np.random.uniform(low=-5, high=5, size=(n_structures, n_beads, 3))
    X = X.ravel()
    X = np.array([[0,0,0],[1.1,0,0]]).ravel()

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

    # space = np.linspace(-1.2,1.2,1000)
    # P['q'].set(1.0)
    # Es = map(lambda x: -P.log_prob(structures=concatenate((X[:3], [x], X[4:]))), space)
    # Es = map(lambda x: -P.log_prob(structures=concatenate((X[:3], [x], X[4:]))), space)
    # P['q'].set(3.06)
    # Es2 = map(lambda x: -P.log_prob(structures=concatenate((X[:3], [x], X[4:]))), space)
    # plt.plot(space, Es, label='q1')
    # plt.plot(space, Es2, label='q2')
    # plt.legend()
    # plt.show()
             
    
