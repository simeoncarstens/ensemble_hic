import unittest
import numpy as np

from ensemble_hic.nonbonded_prior import BoltzmannNonbondedPrior2
from ensemble_hic.nonbonded_prior import AbstractNonbondedPrior2
from ensemble_hic.forcefields import AbstractForceField

class MockAbstractNonbondedPrior2(AbstractNonbondedPrior2):

    def _log_ensemble(self, E):

        return E

    def _log_ensemble_gradient(self, E):

        return 1.0

    def _register_ensemble_parameters(self):
        pass

    def clone(self):
        pass


class MockForcefield(AbstractForceField):

    def energy(self, structure):

        return structure.sum()

    def gradient(self, structure):

        return structure.ravel()
    

class testAbstractNonbondedPrior2(unittest.TestCase):

    def setUp(self):

        self.n_structures = 2
        self.structures = np.array([[[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0]],
                                    [[0.0, 0.0, 2.0],
                                     [0.0, 0.0, 4.0]]])
        self.prior = MockAbstractNonbondedPrior2('bla',
                                                 MockForcefield(np.array([1]),
                                                                42.0),
                                                 self.n_structures)

    def testEvaluate_log_prob(self):

        X = np.random.uniform(size=(self.n_structures, 2, 3))

        res = self.prior._evaluate_log_prob(X.ravel())

        self.assertEqual(res, self.prior.forcefield.energy(structure=X[0]) +
                              self.prior.forcefield.energy(structure=X[1]))

    def testEvaluate_gradient(self):

        X = np.random.uniform(size=(self.n_structures, 2, 3))

        res = self.prior._evaluate_gradient(X.ravel())

        ff = self.prior.forcefield
        self.assertTrue(np.all(res ==  -np.concatenate((ff.gradient(X[0]),
                                                        ff.gradient(X[1])))))


class testBoltzmannNonbondedPrior2(unittest.TestCase):

    def setUp(self):

        self.n_structures = 2
        self.beta = 42.0
        self.prior = BoltzmannNonbondedPrior2('bla',
                                              MockForcefield(np.array([1]),
                                                             42.0),
                                              self.n_structures,
                                              self.beta)

    def testLog_ensemble(self):

        E = 5.0
        res = self.prior._log_ensemble(E)

        self.assertEqual(res, -self.beta * E)

    def testLog_ensemble_gradient(self):

        E = 5.0
        res = self.prior._log_ensemble_gradient(E)

        self.assertEqual(res, -self.beta)

if __name__ == '__main__':

    unittest.main()
