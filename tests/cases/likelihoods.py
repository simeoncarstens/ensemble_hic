import unittest
import numpy as np

from ensemble_hic.likelihoods import Likelihood
from ensemble_hic.forward_models import EnsembleContactsFWM
from ensemble_hic.error_models import PoissonEM

def numgrad(x, E, eps=1e-9):

    import numpy

    res = numpy.zeros(len(x))
    
    E0 = E(x)
    for i in range(len(x)):
        x[i] += eps
        res[i] = (E(x) - E0)
        x[i] -= eps

    return res / eps


class testLikelihood(unittest.TestCase):

    def setUp(self):

        self.n_structures = 3
        self.data_points = np.array([[0, 1, 42],
                                     [0, 2, 23],
                                     [1, 2, 44],
                                     [1, 4, 10],
                                     [0, 4, 54]
                                     ])
        self.contact_distances = np.array([1.0,
                                           0.7,
                                           1.0,
                                           0.4,
                                           1.15,
                                           0.3
                                           ])
        fwm = EnsembleContactsFWM('bla', self.n_structures,
                                  self.contact_distances, self.data_points)
        em = PoissonEM('murks', self.data_points[:,2])
        self.likelihood = Likelihood('lalala', fwm, em, 1.0)

    def testEvaluate_gradient(self):

        for i in range(10):
            X = np.random.uniform(low=-0.6,high=0.6, size=(self.n_structures,
                                                           5, 3))
            alpha = 2.0
            gamma = 3.0
            weights = np.ones(self.n_structures) * alpha
            L = self.likelihood
            res = L._evaluate_gradient(structures=X.ravel(),
                                       norm=alpha,
                                       smooth_steepness=gamma,
                                       weights=weights)
            ng = numgrad(X.ravel(),
                         lambda x: -L._evaluate_log_prob(structures=x,
                                                         norm=alpha,
                                                         smooth_steepness=gamma,
                                                         weights=weights))
            self.assertTrue(np.max(np.fabs(res - ng)) < 1e-3)

if __name__ == '__main__':

    unittest.main()
