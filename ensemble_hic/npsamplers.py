import numpy as np

from abc import ABCMeta, abstractmethod

class AbstractGammaSampler(object):
    '''
    Assumes a uniform prior distribution
    '''
    __metaclass__ = ABCMeta
    
    pdf = None

    def __init__(self, likelihood_name, variable_name):

        self._likelihood_name = likelihood_name
        self._variable_name = variable_name

    @abstractmethod
    def _calculate_shape(self):
        pass

    @abstractmethod
    def _calculate_scale(self):
        pass
    
    def sample(self, state=42):

        rate = self._calculate_rate()        
        shape = self._calculate_shape()
        sample = np.random.gamma(shape) / rate
        
        if sample == 0.0:
            sample += 1e-10

        self.state = sample
        
        return self.state


class NormGammaSampler(AbstractGammaSampler):
    """
    Appropriate for Poisson error model
    """

    def __init__(self):

        super(NormGammaSampler, self).__init__('ensemble_contacts', 'norm')

    def _get_prior(self):

        prior = filter(lambda p: 'norm' in p.parameters, self.pdf.priors)[0]
        
        return prior

    def _check_gamma_prior(self, prior):

        from .gamma_priors import GammaPrior
        
        return isinstance(prior, GammaPrior):

    def _calculate_shape(self):

        prior = self._get_prior()
        if self._check_gamma_prior(prior):
            L = self.pdf.likelihoods[self._likelihood_name]
            lammda = L['lammda'].value
            
            return lammda * L.error_model.data.sum() + prior['shape'].value - 1
        else:
            raise NotImplementedError('Currently, for the scaling parameter, only Gamma priors are supported')

    def _calculate_rate(self):

        prior = self._get_prior()
        L = self.pdf.likelihoods[self._likelihood_name]
        md = L.forward_model(norm=1.0)
        
        return L['lammda'].value * md.sum() + prior['rate'].value
