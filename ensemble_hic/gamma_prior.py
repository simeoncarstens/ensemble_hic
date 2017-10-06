from abc import ABCMeta, abstractmethod

import numpy as np
import scipy

from isd2.pdf.priors import AbstractPrior


class GammaPrior(AbstractPrior):

    __meta__ = ABCMeta

    @abstractmethod
    def __init__(self, name, shape, rate, variable_name):

        from csb.statistics.pdf.parameterized import Parameter

        super(GammaPrior, self).__init__(name)

        self._register('shape')
        self._register('rate')
        self['shape'] = Parameter(shape, 'shape')
        self['rate'] = Parameter(rate, 'rate')

        self._register_variable(variable_name)
        self.update_var_param_types(**{variable_name: Parameter})
        self._set_original_variables()
        
    @abstractmethod
    def _evaluate_log_prob(self, variable):

        shape = self['shape'].value
        rate = self['rate'].value

        return (shape - 1.0) * np.log(variable) - variable * rate


class NormGammaPrior(GammaPrior):

    def __init__(self, shape, rate):

        super(NormGammaPrior, self).__init__('norm_prior', shape, rate, 'norm')

    def _evaluate_log_prob(self, norm):

        return super(NormGammaPrior, self)._evaluate_log_prob(variable=norm)

    def clone(self):

        copy = self.__class__(self['rate'].value, self['shape'].value)

        copy.set_fixed_variables_from_pdf(self)

        return copy
