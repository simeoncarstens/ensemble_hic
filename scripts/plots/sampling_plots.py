import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from isd2.pdf.posteriors import Posterior
    
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.setup_functions import setup_weights
from ensemble_hic.analysis_functions import load_sr_samples

import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}
matplotlib.rc('font', **font)

sys.argv[1] = '/scratch/scarste/ensemble_hic/eser2017/chr4_nozeros_it2_10structures_sn_153replicas/config.cfg'

config_file = sys.argv[1]
settings = parse_config_file(config_file)
n_replicas = int(settings['replica']['n_replicas'])
target_replica = n_replicas
burnin = 5000
n_samples = 7001#int(settings['replica']['n_samples'])
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
x_interval = 15
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

ax = fig.add_subplot(335, facecolor='grey')
def remove_zero_beads(m):
    nm = filter(lambda x: not np.all(np.isnan(x)), m)
    nm = np.array(nm).T
    nm = filter(lambda x: not np.all(np.isnan(x)), nm)
    nm = np.array(nm).T

    return nm

from csb.bio.utils import distance_matrix
vardict = {k: samples[-1,-1].variables[k] for k in samples[-1,-1].variables.keys()
           if k in p.likelihoods['ensemble_contacts'].forward_model.variables}
rec = p.likelihoods['ensemble_contacts'].forward_model(**vardict)
m_mock = np.zeros((n_beads, n_beads)) * np.nan
m_mock[data[:,0], data[:,1]] = rec
m_mock[data[:,1], data[:,0]] = rec
m_mock = remove_zero_beads(m_mock)

m_data = np.zeros((n_beads, n_beads)) * np.nan
m_data[data[:,0],data[:,1]] = data[:,2]
m_data[data[:,1],data[:,0]] = data[:,2]
m_data = remove_zero_beads(m_data)

m_max = np.max([np.max(m_mock[~np.isnan(m_mock)]), np.max(m_data[~np.isnan(m_data)])])
m_mock[0,0] = m_max
m_data[0,0] = m_max
print m_max

ms = ax.matshow(np.log(m_mock+1), cmap=plt.cm.jet, interpolation='nearest')
# ms = ax.matshow(m, cmap=plt.cm.jet)
ax.set_title('rec')
cb = fig.colorbar(ms, ax=ax)
cb.set_label('log(cf)')
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(336, facecolor='grey')
ms = ax.matshow(np.log(m_data+1), cmap=plt.cm.jet, interpolation='nearest')
# ms = ax.matshow(m, cmap=plt.cm.jet)
ax.set_title('data')
cb = fig.colorbar(ms, ax=ax)
cb.set_label('log(cf)')
ax.set_xticks([])
ax.set_yticks([])


ax = fig.add_subplot(334)
hmc_pacc = np.loadtxt(output_folder + 'statistics/mcmc_stats.txt',
                      dtype=float)[:,2 * n_replicas]
plt.plot(hmc_pacc)
plt.xlabel('~# of samples')
plt.ylabel('target HMC dt')

if 'norm' in variables:
    ax = fig.add_subplot(333)
    plt.hist(norms, bins=int(np.sqrt(len(norms))))
    ax.set_xlabel('norm')

ax = fig.add_subplot(337)
E_likelihood = -np.array([L.log_prob(**x.variables)
                          for x in samples[-1,:]])
plt.plot(E_likelihood, label='likelihood', color='blue')
plt.xlabel('rts')
plt.ylabel('target ll')
ax = fig.add_subplot(338)
E_backbone = -np.array([p.priors['backbone_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])
plt.plot(E_backbone, label='backbone', color='green')
plt.xlabel('rts')
plt.ylabel('target bbe')
ax = fig.add_subplot(339)
E_exvol = -np.array([p.priors['nonbonded_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])
plt.plot(E_exvol, label='volume exclusion', color='red')
plt.xlabel('rts')
plt.ylabel('target nbe')

fig.tight_layout()

if save_figures:
    plt.savefig(figures_folder + 'sampling_stats.pdf')
else:
    plt.show()

def calc_Rg_profile(struct, ws):

    from csb.bio.utils import radius_of_gyration

    return np.array([radius_of_gyration(struct[i:i+ws])
                     for i in range(len(struct)-ws)])

windowsize = 20
samples = samples.squeeze()
structs = np.array([x.variables['structures'].reshape(-1, n_beads, 3)
                    for x in samples[-10:]])

profiles = np.array([map(lambda x: calc_Rg_profile(x, windowsize), states) for states
                     in structs])
