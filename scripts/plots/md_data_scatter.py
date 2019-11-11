import sys
import os
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_sr_samples, load_samples_from_cfg_auto

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
n_beads = 308

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains{}{}_{}structures_{}replicas/'.format(it, rep, n_structures, n_replicas)


config_file = sim_path + 'config.cfg'
settings = parse_config_file(config_file)
samples = load_samples_from_cfg_auto(config_file, 30000)
p = make_posterior(settings)
fwm = p.likelihoods['ensemble_contacts'].forward_model
energies = np.array(map(lambda x: -p.log_prob(**x.variables), samples))
map_sample = samples[np.argmin(energies)]
n_structures = fwm.n_structures

def plot_md_d_scatter(ax):
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
    ax.set_xlabel('experimental counts', fontsize=14)
    ax.set_ylabel('back-calculated counts', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles = (handles[1], handles[2], handles[0])
    labels = (labels[1], labels[2], labels[0])
    ax.legend(handles, labels, frameon=False, title='linear distance [beads]:')

    
