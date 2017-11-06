"""
Reconstruction of GB1 and Sh3 from mixed contacts
"""
import os, sys, numpy as np, glob

pypath = os.path.expanduser('~/projects/ensemble_hic/data/proteins/')
os.chdir(pypath)
if not pypath in sys.path: sys.path.insert(0, pypath)

from csb.io import load
from csb.bio.io import StructureParser
from csb.bio.utils import distance_matrix
from csb.statistics.pdf import MultivariateGaussian
from csbplus.bio.dynamics import calc_distances
from scipy.spatial.distance import squareform
from isd import utils
from isd.Distance import Distance
from isd.DataSet import DataSet
from ensemble_hic import kth_diag_indices

def zero_diagonals(a, n):
    for i in range(-n, n+1):
        inds = kth_diag_indices(a, i)
        a[inds] = 0
        
    return a


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

coords  = StructureParser(prot1 + '.pdb').parse().get_coordinates(['CA']) / 4.0
coords2 = StructureParser(prot2 + '.pdb').parse().get_coordinates(['CA']) / 4.0


## size of fake ensemble
ensemble_size = 100
## sigma of Gausian noise added to positions to generate random ensemble
sigma = 0.01
## cutoff from which to determine contact frequencies
cutoff = 1.8

n_beads = len(coords)

if False:
    ## create data by adding Gaussian noise
    g = MultivariateGaussian(mu=coords.ravel(), sigma=sigma * numpy.eye(n_beads*3))
    ensemble1 = g.random(size=ensemble_size).reshape(ensemble_size, n_beads, 3)
    g = MultivariateGaussian(mu=coords2.ravel(), sigma=sigma * numpy.eye(n_beads*3))
    ensemble2 = g.random(size=ensemble_size).reshape(ensemble_size, n_beads, 3)
    dms1 = numpy.array(map(distance_matrix, ensemble1))
    dms2 = numpy.array(map(distance_matrix, ensemble2))
    frequencies1 = numpy.sum(dms1 < cutoff, 0)
    frequencies2 = numpy.sum(dms2 < cutoff, 0)
    summed_frequencies = frequencies1 + frequencies2
elif False:
    ## create Poisson-distributed data
    dm1 = distance_matrix(coords)
    dm2 = distance_matrix(coords2)
    cs1 = ensemble_size * (dm1 < cutoff)
    cs2 = ensemble_size * (dm2 < cutoff)
    summed_frequencies = numpy.random.poisson(cs1 + cs2)
elif False:
    ## create Poisson-distributed data based on probabilites obtained form
    ## non-central Maxwell distribution
    sys.path.append(os.path.expanduser('~/projects/hic/py/'))
    from maxwell import prob_contact
    from scipy.spatial.distance import pdist

    probs1 = numpy.zeros((len(coords), len(coords)))
    probs2 = numpy.zeros((len(coords), len(coords)))
    dm1 = distance_matrix(coords)
    dm2 = distance_matrix(coords2)
    for i in xrange(len(coords)):
        for j in xrange(i+1, len(coords)):
            probs1[i,j] = prob_contact(dm1[i,j], cutoff, sigma)
            probs2[i,j] = prob_contact(dm2[i,j], cutoff, sigma)
    
    probs1[probs1 < 0] = 0.0
    probs2[probs2 < 0] = 0.0

    if prot1 == prot2:
        summed_frequencies = numpy.random.poisson(ensemble_size * probs1)      
    else:
        summed_frequencies = numpy.random.poisson(ensemble_size * (probs1 + probs2))
elif True:    
    from ensemble_hic.forward_models import EnsembleContactsFWM

    suffix = 'fwm_poisson'
    n_structures = 1 if prot1 == prot2 else 2
    data_points = array([[i, j, 0] for i in range(n_beads)
                         for j in xrange(i+1, n_beads)])
    fwm = EnsembleContactsFWM('asdfasdf', n_structures,
                              np.ones(len(data_points)) * 2, data_points)
    structures = np.concatenate((coords,coords2)) if n_structures == 2 else coords
    md = fwm(norm=1.0, smooth_steepness=10,
             structures=structures.ravel(), weights=np.ones(n_structures))
    temp = np.random.poisson(ensemble_size * md)
    summed_frequencies = np.zeros((n_beads, n_beads))
    summed_frequencies[data_points[:,0], data_points[:,1]] = temp
    summed_frequencies[data_points[:,1], data_points[:,0]] = temp
    summed_frequencies = summed_frequencies.astype(int)

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

if True:
    if prot1 == prot2:
        prot2 = 'none'
    with open(os.path.expanduser('~/projects/ensemble_hic/data/proteins/{}_{}/{}.txt'.format(prot1, prot2, suffix)), 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                opf.write('{}\t{}\t{}\n'.format(i, j, summed_frequencies[i,j]))
