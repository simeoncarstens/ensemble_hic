import unittest
import numpy as np

from ensemble_hic.forward_models import EnsembleContactsFWM

class testEnsembleContactsFWM(unittest.TestCase):

    def setUp(self):

        self.n_structures = 2
        self.contact_distances = np.array([2.0, 1.0])
        self.data_points = np.array([[0, 1, 42],
                                     [0, 2, 53]])
        self.fwm = EnsembleContactsFWM('bla', self.n_structures,
                                       self.contact_distances,
                                       self.data_points)

    def testEvaluate(self):

        structures = np.array([[[0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 0.0, 2.0]],
                               [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 3.0],
                                [0.0, 0.0, 3.0]]])

        gamma = 1.0
        norm = 2.0

        md = self.fwm(structures=structures.ravel(),
                      smooth_steepness=gamma,
                      norm=norm)

        ## md[0] should be 2.0 because the sigmoidal function "s" used is
        ## symmetric around the contact distance "d_c" of 2.0 and normalized to
        ## s(d_c) = 0.5. So s(d_c + a) + s(d_c - a) = 1. Then multiply this by
        ## the scaling factor 
        self.assertEqual(md[0], norm * 1.0)

        s = lambda d, gamma, d_c: (gamma * (d_c - d) / \
                                   np.sqrt(1 + (gamma * (d_c - d)) ** 2) + 1) / 2
        contrib1 = s(np.linalg.norm(structures[0,2] - structures[0,0]),
                     gamma, self.contact_distances[1])
        contrib2 = s(np.linalg.norm(structures[1,2] - structures[1,0]),
                     gamma, self.contact_distances[1])
        self.assertEqual(md[1], norm * (contrib1 + contrib2))

if __name__ == '__main__':

    unittest.main()
    
