import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from isd2.pdf.posteriors import Posterior

from csbplus.statmech.wham import WHAM
from csbplus.statmech.dos import DOS
    
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.setup_functions import setup_weights
from ensemble_hic.analysis_functions import load_samples

# config_file = '/scratch/scarste/ensemble_hic/bau2011/GM12878_10structures_s_106replicas_nosphere_optsched3/config.cfg'
config_file = '/scratch/scarste/ensemble_hic/bau2011/GM12878_10structures_s_123replicas_nosphere_specificoptsched/config.cfg'
config_file = '/scratch/scarste/ensemble_hic/bau2011/K562_10structures_s_106replicas_nosphere_optsched3/config.cfg'
# config_file = sys.argv[1]
settings = parse_config_file(config_file)
n_replicas = 106
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
                       dump_interval, burnin=burnin, interval=50)
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



if False:
    import sys, os
    pypath = os.path.expanduser('~/projects/adarex/py')
    if not pypath in sys.path: sys.path.insert(0, pypath)
    from scheduler import Scheduler, RelativeEntropy, SwapRate, SimpleScheduler, SwapRate
    from csbplus.statmech.ensembles import BoltzmannEnsemble
    from cPickle import dump
    
    ensemble = BoltzmannEnsemble(dos=dos)
    entropy  = Scheduler(ensemble, RelativeEntropy(), np.greater)
    
    target = 2.0

    entropy.find_schedule(target, 0., 1., verbose=True)
    
    beta = np.array(entropy.schedule)
    beta[0] = 0.
    beta[-1] = 1.

    step = 5
    tempsched = SimpleScheduler(ensemble, SwapRate(), comparison=np.less)
    pred_swap_rates = [tempsched.eval_criterion(beta[i], beta[i+1])
                       for i in range(len(beta)-1)[::step]]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot((beta[1:] + beta[:-1])[::step] / 2.0, pred_swap_rates)
    ax.set_xlabel('beta')
    ax.set_ylabel('predicted swap rate')
    plt.show()

    with open('/usr/users/scarste/schedules/K562_75replicas.pickle','w') as opf:
        dump(dict(lammda=beta, beta=beta), opf)
