import numpy

from csb.statistics.pdf.parameterized import Parameter

from isd2 import ArrayParameter
from isd2.model.errormodels import AbstractErrorModel


class GaussianEM(AbstractErrorModel):

    def __init__(self, name, data):

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

        super(PoissonEM, self).__init__(name)

        self._register_variable('mock_data', differentiable=True)
        self.data = data
        self.update_var_param_types(mock_data=ArrayParameter)
        self._set_original_variables()

    def _evaluate_log_prob(self, **variables):

        mock_data = variables['mock_data']
        d_counts = self.data

        return -mock_data.sum() + numpy.sum(d_counts * numpy.log(mock_data))

    def _evaluate_gradient(self, **variables):
        pass
    
    def clone(self):

        copy = self.__class__(self.name, self.data)

        copy.set_fixed_variables_from_pdf(self)
        
        return copy


class LognormalEM(AbstractErrorModel):

    def __init__(self, name, data):

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
