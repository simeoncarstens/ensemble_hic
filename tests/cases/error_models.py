import unittest
import numpy as np

from ensemble_hic.error_models import PoissonEM


class testPoissonEM(unittest.TestCase):

    def setUp(self):

        self.data = np.array([3, 4])
        self.em = PoissonEM('bla', self.data)

    def testEvaluate_log_prob(self):

        mock_data = np.array([np.exp(3), np.exp(4)])

        res = self.em._evaluate_log_prob(mock_data=mock_data)

        self.assertEqual(res, -mock_data.sum() + 3 * 3 + 4 * 4)


if __name__ == '__main__':

    unittest.main()

