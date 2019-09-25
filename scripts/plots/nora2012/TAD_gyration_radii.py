import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from csb.bio.utils import rmsd, radius_of_gyration as rog

from ensemble_hic.analysis_functions import load_sr_samples

n_beads = 308

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_fixed_it3_rep3_20structures_309replicas/'
s = load_sr_samples(sim_path + 'samples/', 309, 50001, 1000, 30000)
X = np.array([x.variables['structures'].reshape(20, 308, 3)
              for x in s]) * 53

# sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_nointer_it3_rep3_20structures_309replicas/'
# s = load_sr_samples(sim_path + 'samples/', 309, 50001, 1000, 30000)
# X_nointer = np.array([x.variables['structures'].reshape(20, 308, 3)
#                       for x in s]) * 53

t1 = X[:,:,:107]
t2 = X[:,:,107:]
t1flat = t1.reshape(-1, 107,3)
t2flat = t2.reshape(-1, 201,3)

rgs_tsix = np.array(map(rog, t1flat))
rgs_xist = np.array(map(rog, t2flat))

def plot_rg_heatmap(ax):
    
    from scipy.stats import spearmanr
    print spearmanr(rgs_tsix, rgs_tsix)
    ax.hist2d(rgs_tsix, rgs_xist, bins=np.linspace(150, 300, 50))
    ax.set_xlabel('gyration radius $r_g$ [nm] (Tsix)')
    ax.set_ylabel('gyration radius $r_g$ [nm] (Xist)')
    ax.set_xlim((150, 300))
    ax.set_ylim((150, 300))
    ax.set_aspect('equal')
    
def plot_histograms(ax):

    hargs = dict(bins=np.linspace(150, 350, 50), histtype='stepfilled',
                 normed=True, alpha=0.6)
    ax.hist(rgs_tsix, label='Tsix', color='red', **hargs)
    ax.hist(rgs_xist, label='Xist', color='blue', **hargs)
    for spine in ('top', 'left', 'right'):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(r'radius of gyration $r_g$ [nm]')
    ax.yaxis.set_visible(False)
    # ax.legend(frameon=False)
    
    
