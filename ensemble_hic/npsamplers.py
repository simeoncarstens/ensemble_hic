"""
Samplers for nuisance parameters
"""
import numpy as np

from abc import ABCMeta, abstractmethod

class AbstractGammaSampler(object):

    __metaclass__ = ABCMeta
    
    pdf = None

    def __init__(self, likelihood_name, variable_name):
        """
        Class implementing a sampler which draws from a
        Gamma distribution
        """
        self._likelihood_name = likelihood_name
        self._variable_name = variable_name

    @abstractmethod
    def _calculate_shape(self):
        """
        Calculates the shape of the Gamma distribution

        :returns: shape of Gamma distribution
        :rtype: float > 0
        """
        pass

    @abstractmethod
    def _calculate_rate(self):
        """
        Calculates the rate of the Gamma distribution

        :returns: rate of Gamma distribution
        :rtype: float > 0
        """
        pass
    
    def sample(self, state=42):
        """
        Draws a sample from the Gamma distribution specified
        by a rate and a scale parameter

        :returns: a sample
        :rtype: float
        """
        rate = self._calculate_rate()        
        shape = self._calculate_shape()
        sample = np.random.gamma(shape) / rate
        
        if sample == 0.0:
            sample += 1e-10

        self.state = sample
        
        return self.state


class NormGammaSampler(AbstractGammaSampler):

    def __init__(self):
        """
        Appropriate sampler for the scaling factor when using the usual
        forward model and a Poisson error model and a Gamma prior for
        the scaling factor
        """
        super(NormGammaSampler, self).__init__('ensemble_contacts', 'norm')

    def _get_prior(self):
        """
        Retrieves the prior distribution object associated with the
        scaling factor variable

        :returns: prior distribution object
        :rtype: :class:`.NormGammaPrior`
        """
        prior = filter(lambda p: 'norm' in p.variables, self.pdf.priors.values())[0]
        
        return prior

    def _check_gamma_prior(self, prior):
        """
        Checks whether retrieved prior distribution is in fact a Gamma
        distribution

        :param prior: a prior distribution object
        :type prior: :class:`binf.pdf.priors.AbstractPrior`

        :returns: isn't this self-documenting??
        :rtype: bool
        """
        from .gamma_prior import GammaPrior
        
        return isinstance(prior, GammaPrior)

    def _calculate_shape(self):

        prior = self._get_prior()
        if self._check_gamma_prior(prior):
            L = self.pdf.likelihoods[self._likelihood_name]
            lammda = L['lammda'].value
            
            return lammda * L.error_model.data.sum() + prior['shape'].value
        else:
            raise NotImplementedError('Currently, for the scaling parameter, only Gamma priors are supported')

    def _calculate_rate(self):

        prior = self._get_prior()
        L = self.pdf.likelihoods[self._likelihood_name]
        md = L.forward_model(norm=1.0)
        
        return L['lammda'].value * md.sum() + prior['rate'].value
