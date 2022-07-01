"""
DOS estimation using the nonparametric weighted histogram analysis method (WHAM)
All code in here copyright by Michael Habeck
"""
import numpy as np
import copy

from csb.numeric import log_sum_exp, log, exp

class DOS(object):
    """
    Density of states (DOS)
    """
    def __init__(self, E, s, sort_energies=True, states=None):
        """
        Initialize with a log-density of states (microcanonical entropy)

        @param E: energies at which the DOS is given
        @type E: numpy.ndarray

        @param s: microcanonical entropy (i.e. log(g(E)) where g(E) is the DOS
        @type s: numpy.ndarray
        """
        if len(E) != len(s):
            msg = 'energy and entropy arrays must have equal length'
            raise ValueError(msg)

        self._E = np.array(E)
        self._s = np.array(s)

        if sort_energies and self._E.ndim == 1:
            self._s = self._s[np.argsort(self._E)]
            self._E = np.sort(self._E)

        if states is not None and len(states) != len(self._s):
            msg = 'Number of states must be {}'
            raise ValueError(msg.format(self.size))

        self._states = states

    @property
    def E(self):
        return self._E

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, value):
        if len(value) != len(self._s):
            msg = 'Number of states must be {}'
            raise ValueError(msg.format(self.size))
        self._s = value

    @property
    def log_g(self):
        return self._s

    @property
    def size(self):
        return len(self._s)

    @property
    def states(self):
        return self._states

    def normalize(self):
        """
        Normalizes the density of states
        """
        self.s -= log_sum_exp(self.s)

    def log_Z(self, beta = 1.):
        """
        Logarithm of the partition function
        """
        if len(self.E.shape)>1:
            log_z = - np.dot(beta, self.E.T) + self.log_g
        else:
            log_z = - np.multiply.outer(beta, self.E) + self.log_g

        return log_sum_exp(log_z.T, 0)

    def log_p(self, beta = 1.):

        return - beta * self.E + self.log_g - self.log_Z(beta)

    def compress(self, bins=None):

        ## TODO: works only for 1D DOS

        if bins is None: bins = np.sort(list(set(self.E.tolist())))

        s = np.zeros(len(bins)) + self.s.min()

        for i in range(len(bins)):
            mask = self.E == bins[i]
            if mask.sum() == 0: continue
            s[i] = log_sum_exp(np.compress(mask,self.s))

        return DOS(bins, s)


class PyWHAM(object):
    """
    DOS estimation using the weighted histogram analysis method (WHAM)
    """
    def __init__(self, n_states, n_ensembles):
        """
        @param n_states, n_ensembles: number of states and number of ensembles
        @type n_states, n_ensembles: positive integers
        """
        ## microcanonical entropy
        self.s = np.zeros(n_states)

        ## free energy (log-partition function)
        self.f = np.zeros(n_ensembles)

        ## histograms
        self._H = np.ones(n_states)
        self._N = np.ones(n_ensembles)

        ## for storing log likelihood during DOS estimation
        self.L = []

    @property
    def H(self):
        """
        Histogram
        """
        return self._H
    @H.setter
    def H(self, H):
        self._H[:] = H[:]

    @property
    def N(self):
        """
        Number of samples per ensemble
        """
        return self._N
    @N.setter
    def N(self, N):
        self._N[:] = N[:]

    def stop(self, tol=None):

        return tol is not None and len(self.L) > 1 and \
               abs((self.L[-2]-self.L[-1]) / (self.L[-2]+self.L[-1])) < tol

    def log_likelihood(self, q):

        f = - log_sum_exp((-q + self.s).T, 0)
        L = np.dot(self.H, self.s) + np.dot(self.N, f)

        return -L

    def run(self, q, niter=100, tol=None, verbose=0):
        """
        Run WHAM iterations

        @param q: matrix of reduced energies
        @type q: numpy.ndarray of rank (n_ensembles, n_states)

        @param niter: number of WHAM iterations
        @type niter: positive integer
        """
        H = log(self.H)
        N = log(self.N)

        for i in range(niter):

            ## update free energies

            self.f = - log_sum_exp((-q + self.s).T, 0)

            ## store log likelihood and report on progress

            self.L.append(-(self.N * self.f).sum() - (self.H * self.s).sum())

            ## update density of states and normalize
            
            self.s = H - log_sum_exp((-q.T + self.f + N).T, 0)
            self.s = self.s - log_sum_exp(self.s)

            if verbose and not i % verbose: print i, self.L[-1]

            if self.stop(tol): break

    def update_f(self, q):

        self.f = - log_sum_exp((-q + self.s).T, 0)

    def update_s(self, q):

        self.s = log(self.H) - log_sum_exp((-q.T + self.f + log(self.N)).T, 0)
        self.s = self.s - log_sum_exp(self.s)
