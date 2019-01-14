import sys
import os
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_sr_samples

simlist = ((1, 298, 50001, 30000,  '_it3', '', 1000),
           (5, 218, 50001, 30000,  '_it3', '', 1000),
           (10, 297, 50001, 30000, '', '', 500),
           (20, 309, 50001, 30000, '_fixed_it3', '_rep1', 1000),
           (20, 309, 50001, 30000, '_fixed_it3', '_rep2', 1000),
           (20, 309, 50001, 30000, '_fixed_it3', '_rep3', 1000),
           (30, 330, 32001, 20000, '', '_rep1', 1000),
           (30, 330, 43001, 30000, '', '_rep2', 1000),
           (30, 330, 43001, 30000, '', '_rep3', 1000),
           (40, 330, 33001, 20000, '_it2', '', 1000),
           (40, 330, 33001, 20000, '_it2', '_rep1', 1000),
           (40, 330, 33001, 20000, '_it2', '_rep2', 1000))        

n_structures, n_replicas, n_samples, burnin, it, rep, di = simlist[5]
#n_structures, n_replicas, n_samples, burnin, it, rep, di = simlist[int(sys.argv[1])]
n_beads = 308

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains{}{}_{}structures_{}replicas/'.format(it, rep, n_structures, n_replicas)

# n_replicas = 132
# n_structures = 40
# it = '_it3'
# sim_path = '/scratch/scarste/ensemble_hic/nora2012/15kbbins_bothdomains{}_{}structures_{}replicas/'.format(it, n_structures, n_replicas)
# n_samples = 60001

# n_replicas = 39
# n_structures = 10
# it = ''
# sim_path = '/scratch/scarste/ensemble_hic/nora2012/15kbbins_bothdomains{}_{}structures_{}replicas/'.format(it, n_structures, n_replicas)
# n_samples = 60001

# n_replicas = 116
# n_structures = 10
# it = '_it3'
# sim_path = '/scratch/scarste/ensemble_hic/nora2012/15kbbins_bothdomains{}_{}structures_{}replicas/'.format(it, n_structures, n_replicas)
# n_samples = 60001


config_file = sim_path + 'config.cfg'
settings = parse_config_file(config_file)
samples = load_sr_samples(sim_path + 'samples/', n_replicas, n_samples, di,
                          n_samples-1000)
p = make_posterior(settings)
fwm = p.likelihoods['ensemble_contacts'].forward_model
energies = np.array(map(lambda x: -p.log_prob(**x.variables), samples))
map_sample = samples[np.argmin(energies)]


if False:
    d = fwm.data_points
    md = fwm(**map_sample.variables)

    sys.path.append(os.path.expanduser('~/projects/hic/py/hicisd2/'))
    from mantel import *
    
    m_d = np.zeros((308, 308))
    m_d[d[:,0], d[:,1]] = d[:,2]
    m_d[d[:,1], d[:,0]] = d[:,2]

    m_md = np.zeros((308, 308))
    m_md[d[:,0], d[:,1]] = md
    m_md[d[:,1], d[:,0]] = md

    m_d[m_d == 0.0] = m_d[m_d > 0.0].mean()
    m_md[m_md == 0.0] = m_md[m_md > 0.0].mean()

    n = 1000
    corr = np.corrcoef(m_d.ravel(), m_md.ravel())[0,1]
    random_corrs = np.array([np.corrcoef(m_d.ravel(),
                                         diagonal_permute(m_md).ravel())[0,1]
                             for _ in range(n)])
    p = np.sum(random_corrs > corr) / float(n)



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)


fig, ax = plt.subplots()
ax.set_aspect('equal')
markers = ('*', 's', 'o')
cs = ('k', 'gray', 'lightgray')
sels = [np.where(fwm.data_points[:,1] - fwm.data_points[:,0] == sep)[0]
        for sep in (2,3)]
antisel = np.array([i for i in np.arange(len(fwm.data_points))
                    if not i in [x for y in sels for x in y]])
ax.scatter(fwm.data_points[antisel,2], fwm(**map_sample.variables)[antisel],
           s=12, marker=markers[-1], c=cs[-1], alpha=0.6, label='$|i-j|>3$')
for i, sel in enumerate(sels):
    ax.scatter(fwm.data_points[sel,2], fwm(**map_sample.variables)[sel],
               s=12, marker=markers[i], c=cs[i], alpha=0.6,
               label='$|i-j|={}$'.format(i+2))
maks = np.concatenate((fwm.data_points[:,2], fwm(**map_sample.variables))).max()
ax.plot((0, maks * 1.1), (0, maks * 1.1), ls='--', c='k')
ax.set_xscale('log')
ax.set_yscale('log')
print np.corrcoef(fwm.data_points[:,2], fwm(**map_sample.variables))[0,1]
ax.set_xlim(0.9, maks * 1.1)
ax.set_ylim(0.9, maks * 1.1)
ax.set_xlabel('experimental counts')
ax.set_ylabel('back-calculated counts')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
handles, labels = ax.get_legend_handles_labels()
handles = (handles[1], handles[2], handles[0])
labels = (labels[1], labels[2], labels[0])
ax.legend(handles, labels, frameon=False, title='linear distance [beads]:')
fig.set_size_inches(fig.get_size_inches() * 1.15)
fig.tight_layout()

if not False:
    plt.show()
else:
    path = os.path.expanduser('~/projects/ehic-paper/nmeth/supplementary_information/figures/nora_md_data_scatter/')
    fig.savefig(path + '{}structures{}.svg'.format(n_structures, rep))
    fig.savefig(path + '{}structures{}.pdf'.format(n_structures, rep))

    
