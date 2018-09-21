"""
Classes defining various error models for ensemble-averaged 3C data.
Currently, only the Poisson error model is used and is thus the only
one being tested and documented.
"""
import numpy

from csb.statistics.pdf.parameterized import Parameter

from binf import ArrayParameter
from binf.model.errormodels import AbstractErrorModel


class GaussianEM(AbstractErrorModel):

    def __init__(self, name, data):
        """
        Gaussian error model: deviations between idealized and experimental
        data are penalized quadratically. The strength of the penalty
        is set using the precision parameter, which can be regarded as
        a force constant of a quadratic penalty function. It is a nuisance
        parameter and should be estimated from the data.
        """
        super(GaussianEM, self).__init__(name)
		
        self.data = data

        self._register_variable('mock_data', differentiable=True)
        self._register_variable('precision')
        self.update_var_param_types(mock_data=ArrayParameter, precision=Parameter)
        self._set_original_variables()

    def chisquare_sum(self, mock_data):

        return numpy.sum((mock_data - self.data) ** 2)

    def _evaluate_log_prob(self, **variables):

        mock_data = variables['mock_data']
        precision = variables['precision']

        E = (  0.5 * self.chisquare_sum(mock_data) * precision 
             - 0.5 * len(mock_data) * numpy.log(precision / 2.0 / numpy.pi))

        return -E

    def _evaluate_gradient(self, **variables):

        mock_data = variables['mock_data']
        precision = variables['precision']
                
        return (mock_data - self.data) * precision
        
    def clone(self):

        copy = self.__class__(self.name, self.data)

        copy.set_fixed_variables_from_pdf(self)
        
        return copy


class PoissonEM(AbstractErrorModel):

    def __init__(self, name, data):
        """Poisson error model for contact count / frequency data

        This implements a Poisson distribution as an error model
        for count data. The rates are given by the back-calculated
        counts (the mock data).

        :param name: some name for this object, usually, 'poisson_em'
        :type name: str

        :param data: count / frequency data
        :type data: float
        """
        super(PoissonEM, self).__init__(name)

        self._register_variable('mock_data', differentiable=True)
        self.data = data
        self.update_var_param_types(mock_data=ArrayParameter)
        self._set_original_variables()

    def _evaluate_log_prob(self, mock_data):
        """
        Evaluates the log-probability of the data given the mock data
        
        :param mock_data: back-calculated count / frequency data
        :type mock_data: :class:`numpy.ndarray`

        :returns: log-probablity of the data
        :rtype: float
        """
        d_counts = self.data

        return -mock_data.sum() + numpy.sum(d_counts * numpy.log(mock_data))

    def _evaluate_gradient(self, **variables):
        """
        In theory, this evaluates the gradient of the negative log-probability,
        but I usually hardcode the multiplication of this with the forward
        model Jacobian in Cython (see :mod:`.likelihoods_c`)
        """
        pass
    
    def clone(self):
        """Returns a copy of an instance of this class

        :returns: copy of this object
        :rtype: :class:`.PoissonEM`
        """

        copy = self.__class__(self.name, self.data)

        copy.set_fixed_variables_from_pdf(self)
        
        return copy


class LognormalEM(AbstractErrorModel):

    def __init__(self, name, data):
        """
        Log-normal error model: deviations between idealized and experimental
        data are modeled using a log-normal distribution, assuming that the
        logarithm of the experimental counts follows a normal distribution with
        mean given by the idealized data and precision (inverse variance) treated
        as a nuisance parameter. The latter should be estimated from the data
        """
        super(LognormalEM, self).__init__(name)

        self._register_variable('mock_data', differentiable=True)
        self._register_variable('precision')
        self.data = self.targets = data
        self.update_var_param_types(mock_data=ArrayParameter, precision=Parameter)
        self._set_original_variables()

    def _evaluate_log_prob(self, **variables):

        mock_data = variables['mock_data']
        d_counts = self.data
        precision = variables['precision']
        
        return -0.5 * precision * numpy.sum(numpy.log(mock_data / d_counts) ** 2)

    def _evaluate_gradient(self, **variables):
        pass
    
    def clone(self):

        copy = self.__class__(self.name, self.data)

        copy.set_fixed_variables_from_pdf(self)
        
        return copy
