import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import rmsd, radius_of_gyration as rog

from ensemble_hic.analysis_functions import load_sr_samples

n_structures = 20

before = dict(n_samples = 50001,
              data = 'female_',
              rep = 'fixed_rep1_it4_',
              )

after = dict(n_samples = 50001,
             data = 'female_day2_',
             rep = 'fixed_rep3_')

pos_start = 100378306

def plot_rg_hist(X, ax):

    t1 = X[:,:,:107]
    t2 = X[:,:,107:]
    t1flat = t1.reshape(-1, 107,3)
    t2flat = t2.reshape(-1, 201,3)

    rogs_t1 = np.array(map(rog, t1flat))
    rogs_t2 = np.array(map(rog, t2flat))
    t1_color = 'red'
    t2_color = 'blue'
    ax.hist(rogs_t1, bins=100, label='Tsix TAD', alpha=0.6, color=t1_color,
            histtype='stepfilled', normed=True)
    ax.axvline(rogs_t1.mean(), ls='--', color=t1_color, lw=2)
    ax.hist(rogs_t2, bins=100, label='Xist TAD', alpha=0.6, color=t2_color,
            histtype='stepfilled', normed=True)
    ax.axvline(rogs_t2.mean(), ls='--', color=t2_color, lw=2)
    
    ax.set_xlim((150, 350))
    ax.set_xlabel(r'radius of gyration $r_g$ [nm]')
    ax.set_yticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plot_density(X, ax):
    
    cutoff = 3 * 53
    def density(x):
        dms = squareform(pdist(x))
        return [(dms[i] < cutoff).sum() for i in range(308)]
    densities = np.array([[density(x) for x in y] for y in X])
    f = 3e3 / (4./3. * np.pi * cutoff ** 3)
    density_means = densities.reshape(-1, 308).mean(0) * f
    density_stds = densities.reshape(-1, 308).std(0) * f
    xses = np.arange(len(density_means)) * 3e3 + pos_start
    ax.plot(xses,density_means)
    ax.set_xlabel('genomic position [bp]')
    ax.set_ylabel(r'local density [bp/nm$^3$]')
    ax.axvline(xses[107], ls='--', c='r')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.fill_between(xses, density_means + density_stds,
                    density_means - density_stds, color='lightgray')



def plot_before_hist(ax):
    n_replicas = 241
    simpath = '/scratch/scarste/ensemble_hic/nora2012/{2}bothdomains_{3}{0}structures_{1}replicas/'.format(n_structures, n_replicas, '{}', '{}')
    n_samples = before['n_samples']
    path = simpath.format(before['data'], before['rep'])
    s = load_sr_samples(path + 'samples/', n_replicas, n_samples, 1000,
                        n_samples-30000)
    X = np.array([x.variables['structures'].reshape(n_structures, 308, 3)
                  for x in s]) * 53
    plot_rg_hist(X, ax)

def plot_after_hist(ax):
    n_replicas = 217
    simpath = '/scratch/scarste/ensemble_hic/nora2012/{2}bothdomains_{3}{0}structures_{1}replicas/'.format(n_structures, n_replicas, '{}', '{}')
    n_samples = after['n_samples']
    path = simpath.format(after['data'], after['rep'])
    s = load_sr_samples(path + 'samples/', n_replicas, n_samples, 1000,
                        n_samples-30000)
    X = np.array([x.variables['structures'].reshape(n_structures, 308, 3)
                  for x in s]) * 53
    plot_rg_hist(X, ax)

if False:
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 22}
    plt.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    plot_before_hist(ax1)
    plot_after_hist(ax2)
    plt.gcf().tight_layout()
    plt.show()
