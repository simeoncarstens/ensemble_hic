import unittest
import numpy as np

from ensemble_hic.backbone_prior import BackbonePrior


class testBackbonePrior(unittest.TestCase):

    def setUp(self):

        self.ll = np.array([np.array([0.0, 0.5]),
                            np.array([1.0, 0.0, 0.0])])
        self.ul = np.array([np.array([0.5, 1.0]),
                            np.array([2.0, 2.0, 1.0])])
        self.k_bb = 2.0
        self.n_structures = 2
        self.mol_ranges = np.array([0, 3, 7])

        self.prior = BackbonePrior('bla', self.ll, self.ul, self.k_bb,
                                   self.n_structures, self.mol_ranges)
        self.structures = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.25],
                                    [0.0, 0.0, 4.0],
                                    [0.0, 0.0, 4.5],
                                    [0.0, 0.0, 6.5],
                                    [0.0, 0.0, 8.5]])
        self.structures = np.vstack((self.structures,
                                     [[0.0, 0.0, 2.0],
                                      [0.0, 0.0, 4.0],
                                      [0.0, 0.0, 6.0],
                                      [0.0, 0.0, 7.0],
                                      [0.0, 0.0, 8.0],
                                      [0.0, 0.0, 9.0],
                                      [0.0, 0.0, 10.0]]))
        self.structures = self.structures.reshape(self.n_structures, -1, 3)

    def testSingle_structure_log_prob(self):

        res = self.prior._single_structure_log_prob(self.structures[0,:3],
                                                    self.ll[0],
                                                    self.ul[0])
        self.assertEqual(res, -0.5 * self.k_bb * (0.0625 + 0.25))

    def testSingle_structure_gradient(self):
        
        res = self.prior._single_structure_gradient(self.structures[0,3:],
                                                    self.ll[1],
                                                    self.ul[1])

        grad = np.array([0.0, 0.0,  (0.5 * 0.5) / 0.5,
                         0.0, 0.0, -(0.5 * 0.5) / 0.5,
                         0.0, 0.0, -(1.0 * 2.0) / 2.0,
                         0.0, 0.0,  (1.0 * 2.0) / 2.0])
        self.assertTrue(np.all(res == grad * self.k_bb))

    def testEvaluate_log_prob(self):

        X = self.structures

        res = self.prior._evaluate_log_prob(X.ravel())

        self.assertEqual(res,   self.prior._single_structure_log_prob(X[0,:3],
                                                                      self.ll[0],
                                                                      self.ul[0])
                              + self.prior._single_structure_log_prob(X[0,3:],
                                                                      self.ll[1],
                                                                      self.ul[1])
                              + self.prior._single_structure_log_prob(X[1,:3],
                                                                      self.ll[0],
                                                                      self.ul[0])
                              + self.prior._single_structure_log_prob(X[1,3:],
                                                                      self.ll[1],
                                                                      self.ul[1]))

    def testEvaluate_gradient(self):

        X = self.structures

        res = self.prior._evaluate_gradient(X.ravel())

        grad00 = self.prior._single_structure_gradient(X[0,:3].ravel(),
                                                       self.ll[0],
                                                       self.ul[0])
        grad01 = self.prior._single_structure_gradient(X[0,3:].ravel(),
                                                       self.ll[1],
                                                       self.ul[1])
        grad10 = self.prior._single_structure_gradient(X[1,:3].ravel(),
                                                       self.ll[0],
                                                       self.ul[0])
        grad11 = self.prior._single_structure_gradient(X[1,3:].ravel(),
                                                       self.ll[1],
                                                       self.ul[1])
        self.assertTrue(np.all(res == np.concatenate((grad00, grad01,
                                                      grad10, grad11))))
                                                    
if __name__ == '__main__':

    unittest.main()
