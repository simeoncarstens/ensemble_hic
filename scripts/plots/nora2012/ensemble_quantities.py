import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import rmsd, radius_of_gyration as rog

from ensemble_hic.analysis_functions import load_sr_samples

n_beads = 308

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_fixed_it3_rep3_20structures_309replicas/'
s = load_sr_samples(sim_path + 'samples/', 309, 50001, 1000, 0000)
X = np.array([x.variables['structures'].reshape(20, 308, 3)
              for x in s]) * 53

rogs = np.array([map(rog, x) for x in X])

def plot_avg_rg_trace(ax):

    skip = 5
    scatter_skip = 30 * 5 / skip * 2
    space = np.arange(1, 50001, 20)[::skip]
    ax.plot(space, rogs[1::skip].mean(axis=1), c='black')
    ax.scatter(space[::scatter_skip].repeat(20).reshape(-1,20),
               rogs[1::skip][::scatter_skip],
               alpha=0.2,s=20, color='black')
    
    ax.set_ylabel('radius of gyration r$_g$ [nm]')
    ax.set_xlabel('# of MCMC samples')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

def plot_rg_vs_nstates(ax):
    sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/misc/'))
    from simlist import simulations
    from ensemble_hic.analysis_functions import load_samples_from_cfg_auto
    load_samples = load_samples_from_cfg_auto
    simdata = simulations['nora2012_15kbbins_fixed']
    n_structures = simdata['n_structures']
    common_path = simdata['common_path']
    output_dirs = simdata['output_dirs']
    avg_rgs = []
    std_rgs = []
    he_avg_rgs = []
    he_std_rgs = []
    for i, n in enumerate(n_structures):
        samples = load_samples(common_path + output_dirs[i] + '/config.cfg',
                               burnin=30000)
        X = np.array([s.variables['structures'].reshape(-1,62,3)
                      for s in samples])
        rgs = np.array([map(rog, x) for x in X])
        avg_rgs.append(np.mean(np.mean(rgs, axis=1)))
        std_rgs.append(np.std(np.mean(rgs, axis=1)))
        he_avg_rgs.append(np.mean(rgs.ravel()))
        he_std_rgs.append(np.std(rgs.ravel()))

    ax.errorbar(n_structures, avg_rgs, std_rgs, color='black', label='ensemble\naverage')
    ax.errorbar(np.array(n_structures) + 0.5, he_avg_rgs, he_std_rgs, color='red', label='flattened\nhyperensemble')
    ax.set_xlabel('number of states $n$')
    ax.set_ylabel('radius of gyration $r_g$ [nm]')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

if False:
    fig, ax = plt.subplots()
    plot_rg_hists(ax)
    plt.show()

if False:
    fig, ax = plt.subplots()
    plot_avg_rg_trace(ax)
    plt.show()
