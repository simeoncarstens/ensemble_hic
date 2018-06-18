import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cPickle import dump

from isd2.pdf.posteriors import Posterior

from csbplus.statmech.wham import WHAM
from csbplus.statmech.dos import DOS
    
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.setup_functions import setup_weights
from ensemble_hic.analysis_functions import load_samples

config_file = sys.argv[1]
# config_file = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_40structures/356replicas/config.cfg'

settings = parse_config_file(config_file)
n_replicas = int(settings['replica']['n_replicas'])
target_replica = n_replicas

# params = {'n_samples': 31500,
#           'burnin': 20000,
#           'samples_step': 50,
#           'niter': int(1e6),
#           'tol': 1e-10
#           }

params = {'n_samples': int(sys.argv[2]),
	  'burnin': int(sys.argv[3]),
          'samples_step': int(sys.argv[4]),
          'niter': int(sys.argv[5]),
          'tol': 1e-10
          }

n_samples = min(params['n_samples'], int(settings['replica']['n_samples']))
dump_interval = int(settings['replica']['samples_dump_interval'])

output_folder = settings['general']['output_folder']
if output_folder[-1] != '/':
    output_folder += '/'
n_beads = int(settings['general']['n_beads'])
n_structures = int(settings['general']['n_structures'])
schedule = np.load(output_folder + 'schedule.pickle')

settings['initial_state']['weights'] = setup_weights(settings)
posterior = make_posterior(settings)
if False:
    from ensemble_hic.setup_functions import make_marginalized_posterior
    posterior = make_marginalized_posterior(settings)
p = posterior
variables = p.variables

from ensemble_hic.analysis_functions import load_sr_samples

energies = []
L = p.likelihoods['ensemble_contacts']
data = L.forward_model.data_points
P = p.priors['nonbonded_prior']
sels = []
for i in range(n_replicas):
    print i
    samples = load_sr_samples(output_folder + 'samples/', i+1, n_samples+1,
                              dump_interval,
                              burnin=params['burnin'],
                              )#interval=params['samples_step'])
    sel = np.random.choice(len(samples),
                           int(len(samples) / float(params['samples_step'])),
                           replace=False)
    samples = samples[sel]
    sels.append(sel)
    energies.append([[-L.log_prob(**x.variables) if 'lammda' in schedule else 0,
                      -P.log_prob(structures=x.variables['structures'])
                      if 'beta' in schedule else 0]
                     for x in samples])
energies = np.array(energies)
energies_flat = energies.reshape(np.prod(energies.shape[:2]), 2)

sched = np.array([schedule['lammda'], schedule['beta']])
q = np.array([[(energy * replica_params).sum() for energy in energies_flat]
                 for replica_params in sched.T])
#q = np.dot(energies_flat, sched).T
print q.shape
wham = WHAM(len(energies_flat), n_replicas)
wham.N[:] = len(energies_flat)/n_replicas
wham.run(q, niter=params['niter'], tol=params['tol'], verbose=100)

from csb.numeric import log_sum_exp
dos_2d = DOS(energies_flat, wham.s, sort_energies=False)
#print dos_2d.log_Z(np.array([1.,1.])) - dos_2d.log_Z(np.array([0.,1.]))
#print log_sum_exp(-dos_2d.E.sum(1) + dos_2d.s) - log_sum_exp(-dos_2d.E[:,1] + dos_2d.s)


dos = DOS(energies_flat, wham.s, sort_energies=False)

ana_path = output_folder + 'analysis/'
if not os.path.exists(ana_path):
    os.makedirs(ana_path)
with open(ana_path + 'dos_it{}.pickle'.format(sys.argv[6]), 'w') as opf:
    dump(dos, opf)
with open(ana_path + 'wham_params_it{}.pickle'.format(sys.argv[6]), 'w') as opf:
    dump(params, opf)
with open(ana_path + 'wham_sels_it{}.pickle'.format(sys.argv[6]), 'w') as opf:
    dump(np.array(sels), opf)

if False:
    from csb.numeric import log_sum_exp
    from csb.statistics.rand import sample_from_histogram
    logp = lambda beta: dos.s - beta * dos.E - log_sum_exp(dos.s - beta * dos.E)
    p = lambda beta: np.exp(logp(beta))

    betas = np.load(output_folder + 'analysis/interp_dos_sched.pickle')['beta']
    states = [samples[np.unravel_index(sample_from_histogram(p(beta)), samples.shape)] for beta in betas]
    from cPickle import dump
    dump(states, open('/scratch/scarste/ensemble_hic/hairpin_s/initstates.pickle','w'))


if False:
    from csb.numeric import log_sum_exp
    from csb.statistics.rand import sample_from_histogram
    n_replicas = 298
    path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_it3_1structures_{}replicas/'.format(n_replicas)
    dos = np.load(path + 'analysis/dos.pickle')
    schedule = np.load('/scratch/scarste/ensemble_hic/nora2012/bothdomains_lambdatempering_fromstates_40structures_99replicas/schedule.pickle')
    logp = lambda beta:   dos.s - beta * dos.E[:,0] - dos.E[:,1] \
                        - log_sum_exp(dos.s - beta * dos.E[:,0] - dos.E[:,1])
    p = lambda beta: np.exp(logp(beta))

    lammdas = schedule['lammda']
    indices = [np.unravel_index(sample_from_histogram(p(lammda)),
                                (n_replicas, dos.E.shape[0] / n_replicas))
               for lammda in lammdas]
    indices = np.array(indices).reshape(99, 2)
    wham_params = np.load(path + 'analysis/wham_params.pickle')
    from ensemble_hic.analysis_functions import load_sr_samples
    states = []
    for replica, sample in indices:
        states.append(load_sr_samples(path + 'samples/', replica + 1,
                                      wham_params['n_samples'],
                                      1000, wham_params['burnin'],
                                      wham_params['samples_step'])[sample])
    from cPickle import dump
    import os
    with open(os.path.expanduser('~/projects/ensemble_hic/scripts/nora2012/lambdaonlystates_1structures.pickle'), 'w') as opf:
        dump(states, opf)
