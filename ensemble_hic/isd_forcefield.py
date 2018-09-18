"""
Non-bonded force fields.

PROLSQ uses a quartic repulsion term to penalize particle-particle
clashes.

Copyright by Michael Habeck (2016)
"""
import numpy as np

from ._ensemble_hic import prolsq
from .nblist import NBList
from .core import ctypeproperty, CWrapper, Nominable

class Forcefield(Nominable, CWrapper):
    """Forcefield

    Non-bonded force field enforcing volume exclusion. 
    """
    @ctypeproperty(np.array)
    def k():
        pass

    @ctypeproperty(np.array)
    def d():
        pass

    @ctypeproperty(int)
    def n_types():
        pass

    @property
    def nblist(self):
        return self._nblist

    @nblist.setter
    def nblist(self, value):
        
        self._nblist = value
        if value is not None:
            self.ctype.nblist = value.ctype

    def __init__(self, name):

        self.init_ctype()
        self.set_default_values()

        self.name = name

    def __getstate__(self):

        state = super(Forcefield, self).__getstate__()

        state['nblist'] = state.pop('_nblist')
        state['n_types'] = self.n_types
        state['d'] = self.d
        state['k'] = self.k
        
        return state
    
    def set_default_values(self):

        self.nblist = None
        
        self.enable()

    def is_enabled(self):
        return self.ctype.enabled == 1

    def enable(self, enabled = 1):
        self.ctype.enabled = int(enabled)

    def disable(self):
        self.enable(0)

    def update_list(self, coords):
        """
        Update neighbor list.
        """
        self.ctype.nblist.update(coords.reshape(-1,3),1)
        
    def energy(self, coords, update=True):

        if update: self.update_list(coords)

        return self.ctype.energy(coords.reshape(-1,3), self.types)

    def update_gradient(self, coords, forces):

        return self.ctype.update_gradient(coords, forces, self.types, 1)

    def __str__(self):

        s = '{0}(n_types={1:.2f})'
        
        return s.format(self.__class__.__name__, self.n_types)

    __repr__ = __str__

class PROLSQ(Forcefield):

    def __init__(self, name='PROLSQ'):
        super(PROLSQ, self).__init__(name)

    def init_ctype(self):
        self.ctype = prolsq()
