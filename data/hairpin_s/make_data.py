'''
Makes mock ensemble contact data for a 2D snake (spiral) and a 2D hairpin
'''
import os, numpy
import matplotlib.pyplot as plt
from csb.bio.utils import distance_matrix
from csb.statistics.pdf import MultivariateGaussian
numpy.seterr(all='raise')
from ensemble_hic import kth_diag_indices
numpy.random.seed(42)

def zero_diagonals(a, n):
    b = a.copy()
    for i in range(-n, n+1):
        inds = kth_diag_indices(b, i)
        b[inds] = 0
        
    return b

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

## size of fake ensemble
ensemble_size = 100
## sigma of Gausian noise added to positions to generate random ensemble
sigma = 0.01
## cutoff from which to determine contact frequencies
cutoff = 1.8

n_beads = len(init1)


if True:
    from ensemble_hic.forward_models import EnsembleContactsFWM

    suffix = '_fwm_poisson'
    data_points = array([[i, j, 0] for i in range(n_beads) for j in xrange(i+1, n_beads)])
    fwm = EnsembleContactsFWM('asdfasdf', 2, np.ones(len(data_points)) * 2.3, data_points)
    md = fwm(norm=1.0, smooth_steepness=10,
             structures=concatenate((init1,init2)).ravel(), weights=np.ones(2))
    temp = np.random.poisson(ensemble_size * md)
    summed_frequencies = np.zeros((n_beads, n_beads))
    summed_frequencies[data_points[:,0], data_points[:,1]] = temp
    summed_frequencies[data_points[:,1], data_points[:,0]] = temp
    summed_frequencies = summed_frequencies.astype(int)
    
if False:
    if sigma == 0.01:
        suffix = '_littlenoise'
    elif sigma == 0.05:
        suffix = ''
    else:
        raise
    g = MultivariateGaussian(mu=init1.ravel(), sigma=sigma * numpy.eye(n_beads*3))
    ensemble1 = g.random(size=ensemble_size).reshape(ensemble_size, n_beads, 3)
    g = MultivariateGaussian(mu=init2.ravel(), sigma=sigma * numpy.eye(n_beads*3))
    ensemble2 = g.random(size=ensemble_size).reshape(ensemble_size, n_beads, 3)
    dms1 = numpy.array(map(distance_matrix, ensemble1))
    dms2 = numpy.array(map(distance_matrix, ensemble2))
    frequencies1 = numpy.sum(dms1 < cutoff, 0)
    frequencies2 = numpy.sum(dms2 < cutoff, 0)
    summed_frequencies = frequencies1 + frequencies2

#from misc import kth_diag_indices
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
    with open(os.path.expanduser('~/projects/ensemble_hic/data/hairpin_s/hairpin_s{}.txt'.format(suffix)), 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                opf.write('{}\t{}\t{}\n'.format(i, j, summed_frequencies[i,j]))
