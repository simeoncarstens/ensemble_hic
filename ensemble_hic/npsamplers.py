import numpy as np

from abc import ABCMeta, abstractmethod

class AbstractGammaSampler(object):

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

        scale = self._calculate_scale()        
        shape = self._calculate_shape()

        if False:
            ## find out if variable has a Jeffreys prior
            from hicisd2.priors.jeffreys import JeffreysPrior
            has_jeffreys = False
            Ps = self.pdf.priors
            for P in Ps:
                if self._variable_name in Ps[P].variables:
                        if isinstance(Ps[P], JeffreysPrior):
                            has_jeffreys = True

            ## correct shape
            shape -= has_jeffreys        

        sample = np.random.gamma(shape, scale)
        
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

    def _calculate_shape(self):

        em = self.pdf.likelihoods[self._likelihood_name].error_model
        return 1.0 + em.data.sum()

    def _calculate_scale(self):

        md = self.pdf.likelihoods[self._likelihood_name].forward_model(norm=1.0)
        return 1.0 / md.sum()
