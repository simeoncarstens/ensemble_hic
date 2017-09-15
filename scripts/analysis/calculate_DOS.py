import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from isd2.pdf.posteriors import Posterior

from csbplus.statmech.wham import WHAM
    
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.setup_functions import setup_weights
from ensemble_hic.analysis_functions import load_samples

config_file = '/scratch/scarste/ensemble_hic/bau2011/K562_10structures_s_80replicas_bl0_tempering_nosphere/config.cfg'
# config_file = sys.argv[1]
settings = parse_config_file(config_file)
n_replicas = 80
target_replica = n_replicas
burnin = 20000
n_samples = int(settings['replica']['n_samples'])
dump_interval = int(settings['replica']['samples_dump_interval'])

output_folder = settings['general']['output_folder']
if output_folder[-1] != '/':
    output_folder += '/'
n_beads = int(settings['general']['n_beads'])
n_structures = int(settings['general']['n_structures'])
schedule = np.load(output_folder + 'schedule.pickle')

settings['initial_state']['weights'] = setup_weights(settings)
posterior = make_posterior(settings)
p = posterior
variables = p.variables
L = posterior.likelihoods['ensemble_contacts']
data = L.forward_model.data_points

samples = load_samples(output_folder + 'samples/', n_replicas, n_samples + 1,
                       dump_interval, burnin=70000, interval=400)
L = p.likelihoods['ensemble_contacts']
P = p.priors['nonbonded_prior']
energies = numpy.array([[[-L.log_prob(**x.variables),
                          -P.log_prob(structures=x.variables['structures'])]
                         for x in y] for y in samples])
energies_flat = energies.reshape(np.prod(samples.shape), -1)
sched = np.array([schedule['lammda'], schedule['beta']])
q = numpy.array([[(energy * params).sum() for energy in energies_flat]
                 for params in sched.T])
#q = np.dot(energies_flat, sched).T

wham = WHAM(len(energies_flat), n_replicas)
wham.N[:] = len(energies_flat)/n_replicas
wham.run(q, niter=int(1e6), tol=1e-10, verbose=10)

dos = DOS(energies_flat.sum(1), wham.s, sort_energies=False)
