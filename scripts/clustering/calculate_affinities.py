import os
import sys
import numpy as np
from scipy.spatial.distance import squareform
from ensemble_hic.setup_functions import parse_config_file
from ensemble_hic.analysis_functions import load_samples_from_cfg
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/clustering/'))

config_file = sys.argv[1]
affinities_str = sys.argv[2]
n_processes = int(sys.argv[3])
step = int(sys.argv[4])
settings = parse_config_file(config_file)
output_folder = settings['general']['output_folder']
n_beads = int(settings['general']['n_beads'])
samples = load_samples_from_cfg(config_file)[::step]

if affinities_str == 'wrmsd':
    from clustering_funcs import wrmsd_affinities as tmpaffs
    from ensemble_hic.setup_functions import make_posterior

    p = make_posterior(settings)
    br = p.priors['nonbonded_prior'].forcefield.bead_radii

    affinities = lambda ens, n_processes: tmpaffs(ens, br ** 3, n_processes)

if affinities_str == 'hamming':

    from clustering_funcs import contact_hamming_affinities as tmpaffs
    from ensemble_hic.setup_functions import make_posterior

    p = make_posterior(settings)
    br = p.priors['nonbonded_prior'].forcefield.bead_radii
    n_beads = len(br)
    contact_distances = np.array([[(br[j] + br[i]) * 1.5 * (i != j)
                                   for j in range(n_beads)]
                                  for i in range(n_beads)])
    contact_distances = squareform(contact_distances)

    affinities = lambda ens, n_processes: tmpaffs(ens, contact_distances,
                                                  np.ones(len(contact_distances)))

ens = np.array([sample.variables['structures'].reshape(-1, n_beads, 3)
                for sample in samples])
ens = ens.reshape(-1, n_beads, 3)

affs = squareform(affinities(ens, n_processes=n_processes))

cfolder = output_folder + '/analysis/clustering/{}/'.format(affinities_str)
if not os.path.exists(cfolder):
    os.makedirs(cfolder)
np.save('{}affs.npy'.format(cfolder), affs)
with open('{}settings.pickle'.format(cfolder), 'w') as opf:
    from cPickle import dump
    dump(dict(step=step), opf)
