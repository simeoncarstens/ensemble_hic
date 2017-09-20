import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from isd2.pdf.posteriors import Posterior
    
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.setup_functions import setup_weights
from ensemble_hic.analysis_functions import load_sr_samples

config_file = '/scratch/scarste/ensemble_hic/bau2011/K562_10structures_s_79replicas_bl0_tempering_nosphere_expsched3/config.cfg'
config_file = '/scratch/scarste/ensemble_hic/bau2011/GM12878_10structures_s_106replicas_nosphere_optsched3/config.cfg'
config_file = '/scratch/scarste/ensemble_hic/bau2011/GM12878_10structures_s_123replicas_nosphere_specificoptsched/config.cfg'
#config_file = '/scratch/scarste/ensemble_hic/bau2011/K562_10structures_s_106replicas_nosphere_optsched3/config.cfg'
config_file = '/scratch/scarste/ensemble_hic/hairpin_s/hairpin_s_littlenoise_radius10_2structures_sn_20replicas/config.cfg'
# config_file = sys.argv[1]
settings = parse_config_file(config_file)
n_replicas = 20
target_replica = n_replicas
burnin = 0
n_samples = int(settings['replica']['n_samples'])
dump_interval = int(settings['replica']['samples_dump_interval'])
save_figures = True

output_folder = settings['general']['output_folder']
if output_folder[-1] != '/':
    output_folder += '/'
n_beads = int(settings['general']['n_beads'])
n_structures = int(settings['general']['n_structures'])
schedule = np.load(output_folder + 'schedule.pickle')

settings['initial_state']['weights'] = setup_weights(settings)
posterior = make_posterior(settings)
posterior['lammda'].set(schedule['lammda'][target_replica - 1])
posterior['beta'].set(schedule['beta'][target_replica - 1])
p = posterior
variables = p.variables
L = posterior.likelihoods['ensemble_contacts']
data = L.forward_model.data_points

samples = load_sr_samples(output_folder + 'samples/', n_replicas, n_samples + 1,
                          dump_interval, burnin=burnin)
samples = samples[None,:]
if 'weights' in samples[-1,-1].variables:
    weights = np.array([x.variables['weights'] for x in samples.ravel()])
if 'norm' in samples[-1,-1].variables:
    norms = np.array([x.variables['norm'] for x in samples.ravel()])

figures_folder = output_folder + 'analysis/sampling_plots/'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

## plot sampling statistics
fig = plt.figure()

ax = fig.add_subplot(331)
energies = [np.load(output_folder + 'energies/replica{}.npy'.format(i+1))
            for i in range(len(schedule))]
energies = np.array(energies).T
energies = energies[int(burnin / float(n_samples) * len(energies)):]
plt.plot(energies.sum(1))
plt.xlabel('MC samples?')
plt.ylabel('extended ensemble energy')
        
ax = fig.add_subplot(332)
acceptance_rates = np.loadtxt(output_folder + 'statistics/re_stats.txt',
                              dtype=float)
acceptance_rates = acceptance_rates[-1,1:]
x_interval = 5
xticks = np.arange(0, n_replicas)[::-1][::x_interval][::-1]
plt.plot(np.arange(0.5, n_replicas - 0.5), acceptance_rates, '-o')
plt.xlabel('lambda')
ax.set_xticks(xticks)
lambda_labels = schedule['lammda'][::-1][::x_interval][::-1]
lambda_labels = ['{:.2f}'.format(l) for l in lambda_labels]
ax.set_xticklabels(lambda_labels)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(xticks)
plt.xlabel('beta')
beta_labels = schedule['beta'][::-1][::x_interval][::-1]
beta_labels = ['{:.2f}'.format(l) for l in beta_labels]
ax2.set_xticklabels(beta_labels)
plt.ylabel('acceptance rate')

ax = fig.add_subplot(335, axisbg='grey')
def remove_zero_beads(m):
    nm = filter(lambda x: not numpy.all(numpy.isnan(x)), m)
    nm = numpy.array(nm).T
    nm = filter(lambda x: not numpy.all(numpy.isnan(x)), nm)
    nm = numpy.array(nm).T

    return nm

from csb.bio.utils import distance_matrix
vardict = {k: samples[-1,-1].variables[k] for k in samples[-1,-1].variables.keys()
           if k in p.likelihoods['ensemble_contacts'].forward_model.variables}
rec = p.likelihoods['ensemble_contacts'].forward_model(**vardict)
m = numpy.zeros((n_beads, n_beads)) * numpy.nan
m[data[:,0], data[:,1]] = rec
m[data[:,1], data[:,0]] = rec
m = remove_zero_beads(m)
ms = ax.matshow(numpy.log(m+1), cmap=plt.cm.jet)
ax.set_title('reconstructed')
cb = fig.colorbar(ms, ax=ax)
cb.set_label('contact frequency')

ax = fig.add_subplot(336, axisbg='grey')
m = numpy.zeros((n_beads, n_beads)) * numpy.nan
m[data[:,0],data[:,1]] = data[:,2]
m[data[:,1],data[:,0]] = data[:,2]
m = remove_zero_beads(m)
ms = ax.matshow(numpy.log(m+1), cmap=plt.cm.jet)
ax.set_title('data')
cb = fig.colorbar(ms, ax=ax)
cb.set_label('contact frequency')


ax = fig.add_subplot(334)
hmc_pacc = np.loadtxt(output_folder + 'statistics/mcmc_stats.txt',
                      dtype=float)[:,2 * n_replicas]
plt.plot(hmc_pacc)
plt.xlabel('~# of MC samples')
plt.ylabel('target ensemble HMC timestep')

if 'norm' in variables:
    ax = fig.add_subplot(333)
    plt.hist(norms, bins=int(np.sqrt(len(norms))))
    ax.set_xlabel('norm')

ax = fig.add_subplot(337)
E_likelihood = -np.array([L.log_prob(**x.variables)
                          for x in samples[-1,:]])
plt.plot(E_likelihood, label='likelihood', color='blue')
plt.xlabel('replica transitions')
plt.ylabel('target ensemble log likelihood')
ax = fig.add_subplot(338)
E_backbone = -np.array([p.priors['backbone_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])
plt.plot(E_backbone, label='backbone', color='green')
plt.xlabel('replica transitions')
plt.ylabel('target ensemble backbone energy')
ax = fig.add_subplot(339)
E_exvol = -np.array([p.priors['nonbonded_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])
plt.plot(E_exvol, label='volume exclusion', color='red')
plt.xlabel('replica transitions')
plt.ylabel('target ensemble nonbonded energy')

if save_figures:
    plt.savefig(figures_folder + 'sampling_stats.pdf')
else:
    plt.show()


if False:
    ## determine optimal schedule
    from hicisd2.hicisd2lib import load_samples
    from csbplus.statmech.wham import WHAM
    pypath = os.path.expanduser('~/projects/adarex/py')
    if not pypath in sys.path: sys.path.insert(0, pypath)
    from scheduler import Scheduler, RelativeEntropy, SwapRate, SimpleScheduler
    from csbplus.statmech.dos import IsingDOS, DOS

    samples = load_samples(output_folder + 'samples/', n_replicas,
                           n_samples + 1, dump_interval, burnin=burnin)[5000:]
    flat_samples = samples.ravel()
    sched = numpy.array(zip(schedule['lammda'], schedule['beta']))
    Es = array([[[L.log_prob(**flat_samples[j].variables),
                  P.log_prob(**flat_samples[j].variables)] for i in range(80)]
                for j in range(len(flat_samples))])
    q = Es * sched

    wham = WHAM(samples.shape[0], len(sched))
    wham.N[:] = len(flat_samples)
    wham.run(q.T, niter=int(1e4), tol=1e-10, verbose=10)

    dos = DOS(Es[::len(sched)].sum(2).flatten(), wham.s, sort_energies=False)
    ensemble = BoltzmannEnsemble(dos=dos)
    entropy  = Scheduler(ensemble, RelativeEntropy(), np.greater)
    
    target = 1.0
    
    entropy.find_schedule(target, 0., 1., verbose=True)
    
    beta = np.array(entropy.schedule)
    beta[0] = 0.
    beta[-1] = 1.

