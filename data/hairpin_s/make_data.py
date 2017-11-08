'''
Makes mock ensemble contact data for a 2D snake (spiral) and a 2D hairpin
'''
import os
import numpy

import matplotlib.pyplot as plt

from csb.bio.utils import distance_matrix
from csb.statistics.pdf import MultivariateGaussian

from ensemble_hic import kth_diag_indices
from ensemble_hic.analysis_functions import write_ensemble

numpy.random.seed(42)

def zero_diagonals(a, n):
    b = a.copy()
    for i in range(-n, n+1):
        inds = kth_diag_indices(b, i)
        b[inds] = 0
        
    return b

data_dir = os.path.expanduser('~/projects/ensemble_hic/data/')
## size of fake ensemble
ensemble_size = 100
contact_distance = 8.0

top = numpy.vstack((numpy.arange(7), numpy.zeros(7), numpy.zeros(7)))
right = numpy.array([6, 1, 0])
left = numpy.array([0, 3, 0])

init1 = numpy.hstack((top, right[:,None],
                      top[:,::-1] + numpy.array([0,2,0])[:,None],
                      left[:,None], top + numpy.array([0,4,0])[:,None])).T
init2 = numpy.zeros((11,3))
init2[:,0] = numpy.arange(11)
init2 = numpy.vstack((init2, [10, 1, 0],
                      init2[::-1,:] + numpy.array([0,2,0])[None,:]))

init1 *= 4
init2 *= 4

write_ensemble(init1[None,:], data_dir + 'hairpin_s/snake.pdb')
write_ensemble(init2[None,:], data_dir + 'hairpin_s/hairpin.pdb')

n_beads = len(init1)

from ensemble_hic.forward_models import EnsembleContactsFWM

suffix = '_fwm_poisson'
data_points = array([[i, j, 0] for i in range(n_beads)
                     for j in xrange(i+1, n_beads)])
fwm = EnsembleContactsFWM('asdfasdf', 2,
                          np.ones(len(data_points)) * contact_distance,
                          data_points)
md = fwm(norm=1.0, smooth_steepness=10,
                structures=concatenate((init1,init2)).ravel(), weights=np.ones(2))
temp = np.random.poisson(ensemble_size * md)
summed_frequencies = np.zeros((n_beads, n_beads))
summed_frequencies[data_points[:,0], data_points[:,1]] = temp
summed_frequencies[data_points[:,1], data_points[:,0]] = temp
summed_frequencies = summed_frequencies.astype(int)

if not True:
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    frequencies1 = zero_diagonals(frequencies1, 2)
    ms = ax1.matshow(frequencies1)
    # plt.colorbar()
    plt.title('2D snake')
    fig.colorbar(ms, ax=ax1)
    
    ax2 = fig.add_subplot(132)
    frequencies2 = zero_diagonals(frequencies2, 2)
    ms = ax2.matshow(frequencies2)
    # plt.colorbar()
    plt.title('2D hairpin')
    fig.colorbar(ms, ax=ax2)

    ax3 = fig.add_subplot(133)
    summed_frequencies = zero_diagonals(summed_frequencies, 2)
    ms = ax3.matshow(summed_frequencies)
    # plt.colorbar()
    plt.title('sum')
    fig.colorbar(ms, ax=ax3)

if True:
    with open(data_dir + 'hairpin_s/hairpin_s{}.txt'.format(suffix)), 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                opf.write('{}\t{}\t{}\n'.format(i, j, summed_frequencies[i,j]))
