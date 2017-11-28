"""
Reconstruction of GB1 and SH3 from mixed contacts
"""
import os
import sys
import numpy as np

from csb.bio.io import StructureParser

from ensemble_hic import kth_diag_indices
from ensemble_hic.forward_models import EnsembleContactsFWM

np.random.seed(42)

write_data = False

def zero_diagonals(a, n):
    for i in range(-n, n+1):
        inds = kth_diag_indices(a, i)
        a[inds] = 0
        
    return a

data_dir = os.path.expanduser('~/projects/ensemble_hic/data/proteins/')
ensemble_size = 100
contact_distance = 8.0

prot1 = '1pga'
prot2 = '1shf'

# prot1 = '1ubq'
# prot2 = '2ma1'

# prot1 = '1pga'
# prot2 = '1pga'

# prot1 = '1shf'
# prot2 = '1shf'

# prot1 = '2ma1'
# prot2 = '2ma1'

# prot1 = '1ubq'
# prot2 = '1ubq'

coords  = StructureParser(data_dir + prot1 + '.pdb').parse().get_coordinates(['CA'])
coords2 = StructureParser(data_dir + prot2 + '.pdb').parse().get_coordinates(['CA'])


n_beads = len(coords)

suffix = 'fwm_poisson'
n_structures = 1 if prot1 == prot2 else 2
data_points = array([[i, j, 0] for i in range(n_beads)
                     for j in xrange(i+1, n_beads)])
fwm = EnsembleContactsFWM('asdfasdf', n_structures,
                          np.ones(len(data_points)) * contact_distance,
                          data_points)
structures = np.concatenate((coords,coords2)) if n_structures == 2 else coords
md = fwm(norm=1.0, smooth_steepness=10,
            structures=structures.ravel(), weights=np.ones(n_structures))
temp = np.random.poisson(ensemble_size * md)
summed_frequencies = np.zeros((n_beads, n_beads))
summed_frequencies[data_points[:,0], data_points[:,1]] = temp
summed_frequencies[data_points[:,1], data_points[:,0]] = temp
summed_frequencies = summed_frequencies.astype(int)

if True:
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
    # cb.set_ticks(np.linspace(0, 1, 6))
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
    # cb.set_ticks(np.linspace(0, 1, 6))
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
    # cb.set_ticks(np.linspace(0, 200, 6))
    cb.set_ticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('bead index')
    ax.set_ylabel('bead index')

    fig.tight_layout()
    plt.show()

# from misc import kth_diag_indices
if False:
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    frequencies1 = zero_diagonals(frequencies1, 2)
    ms = ax1.matshow(frequencies1)
    # plt.colorbar()
    plt.title(prot1)
    fig.colorbar(ms, ax=ax1)
    
    ax2 = fig.add_subplot(132)
    frequencies2 = zero_diagonals(frequencies2, 2)
    ms = ax2.matshow(frequencies2)
    # plt.colorbar()
    plt.title(prot2)
    fig.colorbar(ms, ax=ax2)

    ax3 = fig.add_subplot(133)
    summed_frequencies = zero_diagonals(summed_frequencies, 2)
    ms = ax3.matshow(summed_frequencies)
    # plt.colorbar()
    plt.title('sum')
    fig.colorbar(ms, ax=ax3)

if write_data:
    if prot1 == prot2:
        prot2 = 'none'
    with open(data_dir + '{0}_{1}/{0}_{1}_{2}.txt'.format(prot1, prot2, suffix), 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                opf.write('{}\t{}\t{}\n'.format(i, j, summed_frequencies[i,j]))
