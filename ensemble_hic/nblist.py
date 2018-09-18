"""
Wrapper class for neighbor list.

Copyright by Michael Habeck (2016)
"""
from ._ensemble_hic import nblist
from .core import ctypeproperty, CWrapper

class NBList(CWrapper):
    """NBList

    Neighbor list that allows the evaluation of pairwise through-space
    interactions which have a finite range. Particles are sorted into
    cubic cells whose size is equal or larger than the truncation
    threshold of the nonbonded interaction. After assigning particles
    to cells, the evaluation of the pairwise interactions can be
    restricted to the particles that reside in neighboring cells. By
    this trick the evaluation of pairwise interactions no longer has
    a complexity that is quadratic in the number of particles, but
    grows only linearly with the size of the system.
    """
    @ctypeproperty(float)
    def cellsize():
        """
        Edge length of the cubic cells.
        """
        pass

    @ctypeproperty(int)
    def n_cells():
        """
        Number of cubic cells in each spatial direction such that the
        total number of cells is 'n_cells^3'.
        """
        pass

    @ctypeproperty(int)
    def n_per_cell():
        """
        Maximum number of particles that fits into a cell. 
        """
        pass

    @ctypeproperty(int)
    def n_atoms():
        """
        Number of particles. 
        """
        pass
    
    def __init__(self, cellsize, n_cells, n_per_cell, n_particles):
        """NBList

        Initialize a neighbor list that allows the computation of pairwise
        interactions in linear computational complexity.
        
        Parameters
        ----------

        cellsize :
          length of a single cell

        n_cells :
          number of cells in one dimension
          
        n_per_cell :
          max. no. of atoms assignable to one cell

        n_particles :
          number of particles
          
        """
        self.init_ctype()

        self.cellsize   = cellsize
        self.n_per_cell = n_per_cell
        self.n_atoms    = n_particles

        ## do this only at the very end
        
        self.n_cells    = n_cells
        
        self.enable()

    def init_ctype(self):
        self.ctype = nblist()              

    def enable(self, enable = 1):
        self.ctype.enabled = int(enable)

    def disable(self):
        self.enable(0)

    def is_enabled(self):
        return self.ctype.enabled == 1

    def update(self, universe, update_box=True):
        """
        Update the neighbor list.

        Parameters
        ----------

        universe :
          Universe containing all particles whose pairwise interactions
          will be evaluated.

        update_box : boolean
          By toggling the flag, we can switch off the adaption of the
          cell grid (i.e. the origin of the grid)
          
        """
        self.ctype.update(universe.coords, int(update_box))

    def __setstate__(self, state):

        n_cells = state.pop('n_cells')

        super(NBList, self).__setstate__(state)

        self.n_cells = n_cells
        self.enable()
        
    def __str__(self):

        s = '{0}({1},{2},{3},{4})'

        return s.format(self.__class__.__name__,
                        self.cellsize, self.n_cells,
                        self.n_per_cell, self.n_atoms)

    __repr__ = __str__

