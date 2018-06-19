"""
Forcefields describing non-bonded interactions
"""
import numpy as np

from abc import abstractmethod


class AbstractForceField(object):
    
    def __init__(self, bead_radii, force_constant):
        """
        A purely repulsive forcefield for non-bonded interactions with a potential
        in which pairwise distances closer than the sum of two bead radii are penalized
        quadratically.

        :param bead_radii: list of bead radii
        :type bead_radii: list-like

        :param force_constant: force constant
        :type force constant: float
        """
        self._bead_radii = bead_radii
        self._bead_radii2 = bead_radii * bead_radii
        self._force_constant = force_constant

    @abstractmethod
    def energy(self, structure):
        """
        Evaluates the potentital energy of a structure

        :param structure: coordinates of a structure
        :type structure: :class:`numpy.ndarray`

        :returns: potential energy of a structure
        :rtype: float
        """
        pass

    @abstractmethod
    def gradient(self, structure):
        """
        Evaluates the energy gradient for a structure

        :param structure: coordinates of a structure
        :type structure: :class:`numpy.ndarray`

        :returns: gradient vector
        :rtype: :class:`numpy.ndarray`
        """
        pass

    @property
    def bead_radii(self):
        """
        Bead radii

        :returns: list of bead radii
        :rtype: list-like of floats; length: # beads
        """
        return self._bead_radii
    @bead_radii.setter
    def bead_radii(self, value):
        """
        Sets bead radii

        :param value: list of bead radii
        :type value: :class:`numpy.ndarray`
        """
        self._bead_radii = value

    @property
    def bead_radii2(self):
        """
        Squared bead radii

        :returns: list of squared bead radii
        :rtype: list-like
        """
        return self._bead_radii2
    
    @property
    def force_constant(self):
        """
        Force constant

        :returns: force constant
        :rtype: float
        """
        return self._force_constant
    @force_constant.setter
    def force_constant(self, value):
        """
        Sets force constant

        :param value: new force constant
        :type value: float
        """
        self._force_constant = value

class ForceField(AbstractForceField):
    
    def energy(self, structure):
        """
        Cython implementation of the potential energy

        :param structure: coordinates of a structure
        :type structure: :class:`numpy.ndarray`

        :returns: potential energy of a structure
        :rtype: float
        """
        from ensemble_hic.forcefield_c import forcefield_energy
        
        E = forcefield_energy(structure, self.bead_radii,
                              self.bead_radii2,
                              self.force_constant)

        return E
    
    def gradient(self, structure):
        """
        Cython implementation of the energy gradient

        :param structure: coordinates of a structure
        :type structure: :class:`numpy.ndarray`

        :returns: gradient vector
        :rtype: :class:`numpy.ndarray`
        """        
        from ensemble_hic.forcefield_c import forcefield_gradient
        
        grad = forcefield_gradient(structure, self.bead_radii,
                                   self.bead_radii2,
                                   self.force_constant)
        
        return grad
    
def make_universe(n_beads, L=100):
        
    from isd.Atom import Atom
    from isd.Universe import Universe

    coords = np.random.random((n_beads, 3)) * L
    beads  = [Atom('CA', index=i, x=coords[i]) 
              for i in range(n_beads)]

    return Universe(beads)


class VolumeExclusion(object):

    def __init__(self, universe, bead_types=None):
        
        self._bead_types = bead_types

        self._universe = universe
        self._forcefield = self.make_forcefield(universe)

    @property
    def forcefield(self):
        return self._forcefield

    def make_forcefield(self, universe):

        ## create force field and non-bonded list, assign
        ## the same atom type to all beads

        from isd.prolsq import PROLSQ
        from isd.NBList import NBList

        n_beads    = self._universe.n_atoms
        forcefield = PROLSQ()
        forcefield.n_types = n_beads
        nblist     = NBList(1.0 + 1e-3, 90, n_beads, n_beads)
        forcefield.nblist = nblist

        for i in range(n_beads):
            self._universe.atoms[i].type = i
        self._universe.set_types()
        self._universe.set_connectivity()

        ## set forcefield parameters
        ## will be modified later
        n_types = n_beads
        d = np.ones((n_types, n_types),'d')
        k = np.ones((n_types, n_types),'d')

        forcefield.n_types = n_types
        forcefield.d = d
        forcefield.k = k

        universe.set_types()

        return forcefield

    def energy(self, coords, update=1):

        self._universe.X[:,:] = coords[:,:]
        return self.forcefield.energy(self._universe, update)

    def gradient(self, coords, update=1):

        self._universe.X[:,:] = coords
        self._universe.gradient[:,:] = 0.

        self.forcefield.update_gradient(self._universe, update)
        E = self.forcefield.ctype.update_gradient(self._universe.ctype, 1)

        return self._universe.gradient.copy()


class NBLForceField(AbstractForceField):
    
    def __init__(self, bead_radii, force_constant):
        """
        A purely repulsive forcefield for non-bonded interactions with a potential
        in which pairwise distances closer than the sum of two bead radii are penalized
        quadratically. 

        This implementation uses a non-bonded list written by Michael Habeck.
        It makes evaluating the energy / gradient linear instead quadratic in the
        number of beads.

        :param bead_radii: list of bead radii
        :type bead_radii: list-like

        :param force_constant: force constant
        :type force constant: float
        """

        super(NBLForceField, self).__init__(bead_radii, force_constant)

        self.n_beads = len(bead_radii)
        self._isdff = self._make_volume_exclusion()
        self.set_connectivity()
        self.bead_radii = bead_radii
        self.force_constant = force_constant

    def _make_volume_exclusion(self):

        return VolumeExclusion(make_universe(n_beads=self.n_beads, L=100.0))

    def set_connectivity(self):

        temp = self._isdff._universe.connectivity
        temp = np.ones(temp.shape) - np.eye(temp.shape[0])
        self._isdff._universe.set_connectivity(temp)

    @property
    def bead_radii(self):
        return self._bead_radii
    @bead_radii.setter
    def bead_radii(self, value):
        self._bead_radii = value
        temp = self._isdff.forcefield.d
        temp *= 0.0
        temp += np.add.outer(self._bead_radii, self._bead_radii)
        self._isdff.forcefield.d = temp
        self._isdff.forcefield.nblist.cellsize = temp.max() + 1e-3

    @property
    def force_constant(self):
        return self._force_constant
    @force_constant.setter
    def force_constant(self, value):
        self._force_constant = value
        temp = self._isdff.forcefield.k
        temp *= 0.0
        temp += self._force_constant
        self._isdff.forcefield.k = temp

    def energy(self, structure):

        return self._isdff.energy(structure.reshape(-1,3).astype(np.double))

    def gradient(self, structure):

        return self._isdff.gradient(structure.reshape(-1,3).astype(np.double)).ravel()



if __name__ == '__main__':

    import os
    radii = np.loadtxt(os.path.expanduser('~/projects/ensemble_hic/data/rao2014/chr16_radii.txt'))[:101]
    n_beads = len(radii)
    k = 11.4
    from ensemble_hic.setup_functions import make_elongated_structures
    X = make_elongated_structures(radii, 1) * 0.6
    res = []
    for i in range(500):
        X = np.random.normal(scale=15, size=X.shape)

        oldFF = ForceField(radii, k)
        oldE = oldFF.energy(structure=X.reshape(-1,3))
        
        newFF = NBLForceField(radii, k)
        newE = newFF.energy(structure=X)

        res.append(newE-oldE)

    print "fraction of unequal energies: ", len(filter(lambda x: abs(x) > 0.1, res))/float(len(res))
