"""
Create simulated protein contact data
"""
import os
import sys
import numpy as np

from csb.bio.io import StructureParser

from ensemble_hic import kth_diag_indices
from ensemble_hic.forward_models import EnsembleContactsFWM

np.random.seed(42)

## Set this to true to actually write data files
## False is for testing purposes
write_data = False

def zero_diagonals(a, n):
    for i in range(-n, n+1):
        inds = kth_diag_indices(a, i)
        a[inds] = 0
        
    return a

## adapt path to a directory containing PDB files of proteins of interest
data_dir = os.path.expanduser('~/projects/ensemble_hic/data/proteins/')
## number of simulated copies per protein
ensemble_size = 100
## distance below which to CA atoms are defined to be in contact
contact_distance = 8.0

prot1 = '1pga'
prot2 = '1shf'

## parse PDB files
coords  = StructureParser(data_dir + prot1 + '.pdb').parse().get_coordinates(['CA'])
coords2 = StructureParser(data_dir + prot2 + '.pdb').parse().get_coordinates(['CA'])

## create forward model object 
n_beads = len(coords)
suffix = 'fwm_poisson'
n_structures = 1 if prot1 == prot2 else 2
data_points = np.array([[i, j, 0] for i in range(n_beads)
                        for j in xrange(i+1, n_beads)])
fwm = EnsembleContactsFWM('asdfasdf', n_structures,
                          np.ones(len(data_points)) * contact_distance,
                          data_points)

## calculate idealized (noise-free) simulated data
structures = np.concatenate((coords,coords2)) if n_structures == 2 else coords
md = fwm(norm=1.0, smooth_steepness=10,
         structures=structures.ravel(), weights=np.ones(n_structures))

## add Poisson noise to idealized data
temp = np.random.poisson(ensemble_size * md)

## make square contact frequency matrix
summed_frequencies = np.zeros((n_beads, n_beads))
summed_frequencies[data_points[:,0], data_points[:,1]] = temp
summed_frequencies[data_points[:,1], data_points[:,0]] = temp
summed_frequencies = summed_frequencies.astype(int)

if False:
    ## visualize single-protein contact matrices and the
    ## final simulated contact frequency matrix
    from scipy.spatial.distance import squareform
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fwm_ss = EnsembleContactsFWM('asdfasdf', 1,
                                 np.ones(len(data_points)) * contact_distance,
                                 data_points)
    fig = plt.figure()
    ax = fig.add_subplot(131)
    md_ss = fwm_ss(norm=1.0, smooth_steepness=10,
                   structures=coords.ravel(), weights=np.ones(1))
    m = ax.matshow(squareform(md_ss))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(m, cax=cax)
    cb.set_clim((0,1))
    cb.set_ticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('bead index')
    ax.set_ylabel('bead index')
    ax = fig.add_subplot(132)
    md_ss = fwm_ss(norm=1.0, smooth_steepness=10,
                   structures=coords2.ravel(), weights=np.ones(1))
    m = ax.matshow(squareform(md_ss))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(m, cax=cax)
    cb.set_clim((0,1))
    cb.set_ticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('bead index')
    ax.set_ylabel('bead index')
    ax = fig.add_subplot(133)
    m = ax.matshow(summed_frequencies)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(m, cax=cax)
    cb.set_clim((0,200))
    cb.set_ticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('bead index')
    ax.set_ylabel('bead index')

    fig.tight_layout()
    plt.show()

if write_data:
    ## write contact frequencies to text file used as input for our structure
    ## calculation code
    if prot1 == prot2:
        prot2 = 'none'
    ## please adapt path to your liking
    opfpath = data_dir + '{0}_{1}/{0}_{1}_{2}.txt'.format(prot1, prot2, suffix)
    with open(opfpath, 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                opf.write('{}\t{}\t{}\n'.format(i, j, summed_frequencies[i,j]))

if False:
    ## prepare matrix in correct format for PGS (Alber lab)

    ## assume CA atoms represent TADs of 1mb size
    TAD_size = 1000000

    ## turn contact frequency matrix to probability matrix and
    summed_frequencies = summed_frequencies.astype(float)
    summed_frequencies /= summed_frequencies.max()
    ## set side diagonals to 1.0
    summed_frequencies[kth_diag_indices(summed_frequencies, 1)] = 1.0
    summed_frequencies[kth_diag_indices(summed_frequencies, -1)] = 1.0

    ## write files in appropriate format for the PGS code
    ## (https://github.com/alberlab/pgs/)
    ## please adapt paths to your liking
    oppath = os.path.expanduser('~/projects/mypgs/data/proteins/')
    with open(oppath + 'prob_matrix_for_alber_final.txt'), 'w') as opf:
        for i, line in enumerate(summed_frequencies):
            opf.write('chr1\t{}\t{}'.format(i * TAD_size, (i + 1) * TAD_size))
            for x in line:
                opf.write('\t{:.6f}'.format(x))
            opf.write('\n')

    with open(oppath + 'TADs_for_alber.txt'), 'w') as opf:
        for i in range(len(summed_frequencies)):
            opf.write('chr1\t{}\t{}\tdomain\n'.format(i * TAD_size, (i+1) * TAD_size))
