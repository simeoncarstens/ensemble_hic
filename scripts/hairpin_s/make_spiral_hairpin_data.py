'''
Makes mock ensemble contact data for a 2D snake (spiral) and a 2D hairpin
'''
import os, numpy
import matplotlib.pyplot as plt
from csb.bio.utils import distance_matrix
from csb.statistics.pdf import MultivariateGaussian
from protlib import writeGNMtraj
numpy.seterr(all='raise')

def zero_diagonals(a, n):
    for i in range(-n, n+1):
        inds = kth_diag_indices(a, i)
        a[inds] = 0
        
    return a


if True:
    top = numpy.vstack((numpy.arange(7), numpy.zeros(7), numpy.zeros(7)))
    right = numpy.array([6, 1, 0])
    left = numpy.array([0, 3, 0])
if False:
    top = numpy.vstack((numpy.arange(4), numpy.zeros(4), numpy.zeros(4)))
    right = numpy.array([3, 1, 0])
    left = numpy.array([0, 3, 0])

init1 = numpy.hstack((top, right[:,None], top[:,::-1] + numpy.array([0,2,0])[:,None], left[:,None], top + numpy.array([0,4,0])[:,None])).T
if True:
    init2 = numpy.zeros((11,3))
    init2[:,0] = numpy.arange(11)
    init2 = numpy.vstack((init2, [10, 1, 0], init2[::-1,:] + numpy.array([0,2,0])[None,:]))
if False:
    init2 = numpy.zeros((6,3))
    init2[:,0] = numpy.arange(6)
    init2 = numpy.vstack((init2, [5, 1, 0], init2[::-1,:] + numpy.array([0,2,0])[None,:], numpy.array([-1, 2, 0])))
if True:
    writeGNMtraj(numpy.array([init1]), '/tmp/out1.pdb')
    writeGNMtraj(numpy.array([init2]), '/tmp/out2.pdb')

## size of fake ensemble
ensemble_size = 10000
## sigma of Gausian noise added to positions to generate random ensemble
sigma = 0.05
## cutoff from which to determine contact frequencies
cutoff = 1.8

n_beads = len(init1)

g = MultivariateGaussian(mu=init1.ravel(), sigma=sigma * numpy.eye(n_beads*3))
ensemble1 = g.random(size=ensemble_size).reshape(ensemble_size, n_beads, 3)
g = MultivariateGaussian(mu=init2.ravel(), sigma=sigma * numpy.eye(n_beads*3))
ensemble2 = g.random(size=ensemble_size).reshape(ensemble_size, n_beads, 3)
dms1 = numpy.array(map(distance_matrix, ensemble1))
dms2 = numpy.array(map(distance_matrix, ensemble2))
frequencies1 = numpy.sum(dms1 < cutoff, 0)
frequencies2 = numpy.sum(dms2 < cutoff, 0)
summed_frequencies = frequencies1 + frequencies2

from misc import kth_diag_indices
if True:
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
    with open(os.path.expanduser('~/projects/hic/py/hicisd2/ensemble_scripts/toy_test/spiral_hairpin_data_{}beads_littlenoise.txt'.format(n_beads)), 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                opf.write('{}\t{}\t{}\n'.format(i, j, summed_frequencies[i,j]))
