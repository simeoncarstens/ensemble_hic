import numpy, sys, os
import matplotlib.pyplot as plt
from csb.bio.utils import distance_matrix

from copy import deepcopy
from protlib import writeGNMtraj
numpy.seterr(all='raise')
this_path = os.path.expanduser('~/projects/hic/py/hicisd2/ensemble_scripts/toy_test/')
from isd2.samplers.gibbs import GibbsSampler
from isd2.samplers import ISDState
from misc import make_posterior, make_subsamplers

n_structures = 2
n_beads = 23

X = numpy.random.normal(size=n_beads * n_structures * 3, scale=5)

state = ISDState({'structures': X,
                  'weights': numpy.ones(n_structures) * 10
                })

smooth_steepness = 2.0
norm = 1.0

posterior = make_posterior(n_structures, this_path + 'spiral_hairpin_data.txt',
                           n_beads=23, smooth_steepness=10)
subsamplers = make_subsamplers(posterior, state)
gips = GibbsSampler(pdf=posterior, state=state, subsamplers=subsamplers)

outpath = '/tmp/enshic/'
os.system('mkdir '+outpath)

structures_hmc = gips._subsamplers['structures']
weights_hmc = gips._subsamplers['weights']
samples = []
for i in range(5000):
    samples.append(deepcopy(gips.sample()))
    if i % 10 == 0: 
        print i
        print structures_hmc.acceptance_rate
        print weights_hmc.acceptance_rate
        # print samples[-1].variables['weights']
        print

Es = map(lambda x: -posterior.log_prob(**x.variables), samples)

