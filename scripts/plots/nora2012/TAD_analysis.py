import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import rmsd, radius_of_gyration as rog

from ensemble_hic.analysis_functions import load_sr_samples

simlist = ((1, 298, 50001, 30000,  '_it3', '', 1000),
           (5, 218, 50001, 30000,  '_it3', '', 1000),
           (10, 297, 50001, 30000, '', '', 500),
           (20, 309, 50001, 30000, '_it3', '', 1000),
           (20, 309, 50001, 30000, '_it3', '_rep3', 1000),
           (20, 309, 50001, 30000, '_it3', '_rep4', 1000),
           (30, 330, 32001, 20000, '', '_rep1', 1000),
           (30, 330, 43001, 30000, '', '_rep2', 1000),
           (30, 330, 43001, 30000, '', '_rep3', 1000),
           (40, 330, 33001, 20000, '_it2', '', 1000),
           (40, 330, 33001, 25000, '_it2', '_rep1', 1000),
           (40, 330, 33001, 20000, '_it2', '_rep2', 1000))

n_structures, n_replicas, n_samples, burnin, it, rep, di = simlist[-9]
#n_structures, n_replicas, n_samples, burnin, it, rep, di = simlist[int(sys.argv[1])]
n_beads = 308

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains{}{}_{}structures_{}replicas/'.format(it, rep, n_structures, n_replicas)

s = load_sr_samples(sim_path + 'samples/', n_replicas, n_samples, di, burnin)
X = np.array([x.variables['structures'].reshape(n_structures, 308, 3)
              for x in s]) * 53
Xflat = X.reshape(-1,308,3)

pos_start = 100378306

t1 = X[:,:,:107]
t2 = X[:,:,107:]
t1flat = t1.reshape(-1, 107,3)
t2flat = t2.reshape(-1, 201,3)

if False:
    ## gyration radius histograms
    rogs_t1 = np.array(map(rog, t1flat))
    rogs_t2 = np.array(map(rog, t2flat))
    axes[0,0].hist(rogs_t1, bins=100, label='Tsix TAD', alpha=0.6, color='red')
    axes[0,0].hist(rogs_t2, bins=100, label='Xist TAD', alpha=0.6, color='green')
    axes[0,0].legend(frameon=False)
    axes[0,0].set_xlabel(r'$r_{gyr}$ [nm]')
    axes[0,0].set_yticks(())
    axes[0,0].spines['top'].set_visible(False)
    axes[0,0].spines['right'].set_visible(False)
    axes[0,0].spines['left'].set_visible(False)

if False:
    sub = np.random.choice(len(Xflat), int(len(Xflat)/10))
    def f(x):
        dms = squareform(pdist(x))# < 3 * 53
        return array([diag(fliplr(dms), i).mean() for i in range(-307,308)])
    profiles = np.array(map(f, Xflat[sub]))
    profiles_mean = profiles.mean(0)
    profiles_std = profiles.std(0)
    xses = np.arange(len(profiles_mean)) * 3e3 + pos_start
    axes[0,1].plot(xses, profiles_mean)
    axes[0,1].fill_between(xses, profiles_mean + profiles_std,
                           profiles_mean - profiles_std, color='lightgray')
    axes[0,1].set_xlabel('genomic position [bp]')
    #axes[0,1].set_xticks(())
    axes[0,1].legend(frameon=False)
    axes[0,1].spines['top'].set_visible(False)
    axes[0,1].spines['right'].set_visible(False)
    axes[0,1].spines['left'].set_visible(True)
    #axes[0,1].set_visible(False)
    #axes[0,1].set_ylim((0,0.2))
    #axes[0,1].axvline(xses[107] * 2, ls='--', c='r')
    axes[0,1].set_ylabel('cross-diagonal\ncontact count')
    #axes[0,1].set_yticks(())

if False:
    cutoff = 3 * 53
    def density(x):
        dms = squareform(pdist(x))
        return [(dms[i] < cutoff).sum() for i in range(308)]
    densities = np.array([[density(x) for x in y] for y in X])
    density_means = densities.reshape(-1, 308).mean(0) * 3e3 / (4./3. * np.pi * cutoff ** 3)
    density_stds = densities.reshape(-1, 308).std(0) * 3e3 / (4./3. * np.pi * cutoff ** 3)
    xses = np.arange(len(density_means)) * 3e3 + pos_start
    axes[1,0].plot(xses,density_means)
    axes[1,0].set_xlabel('genomic position [bp]')
    axes[1,0].set_ylabel(r'local density [bp/nm$^3$]')# ($d_c={}$ [nm])'.format(cutoff))
    axes[1,0].axvline(xses[107], ls='--', c='r')
    axes[1,0].spines['top'].set_visible(False)
    axes[1,0].spines['right'].set_visible(False)
    axes[1,0].spines['left'].set_visible(True)
    axes[1,0].fill_between(xses, density_means + density_stds,
                           density_means - density_stds, color='lightgray')

if False:
    window_size = 5
    local_rogs = np.array([[[rog(x[i:i+window_size]) for i in range(0, 308-window_size)]
                            for x in y] for y in X[::10]])
    means = local_rogs.reshape(np.prod(local_rogs.shape[:2]), -1).mean(0)
    stds = local_rogs.reshape(np.prod(local_rogs.shape[:2]), -1).std(0)
    xses = np.arange(len(density_means) - window_size) * 3e3 + pos_start
    axes[1,1].plot(xses, means)
    axes[1,1].set_xlabel('genomic position')
    axes[1,1].set_ylabel(r'local $r_{gyr}$ [nm]')
    axes[1,1].axvline(xses[107], ls='--', c='r')
    axes[1,1].spines['top'].set_visible(False)
    axes[1,1].spines['right'].set_visible(False)
    axes[1,1].fill_between(xses, means + stds, means - stds, color='lightgray')
    axes[1,1].set_xticks(())

if True:
    def plot_TADcm_hist(ax):
        ds = np.array([np.linalg.norm(t1flat[i].mean(0) - t2flat[i].mean(0))
                       for i in range(len(Xflat))])
        ax.hist(ds, bins=50, color='gray')
        ax.set_xlim(0, 600)
        ax.set_xlabel('TAD center of mass distances [nm]')
        ax.axvline(np.mean(map(rog, Xflat)), ls='--', c='black')
        for spine in ('top', 'left', 'right'):
            ax.spines[spine].set_visible(False)
        ax.set_yticks(())

if False:
    itcs = np.array([(squareform(pdist(x))[:107,107:] < 1.3 * 53 * 6).sum()
                     for x in Xflat])
    axes[0,1].hist(itcs, bins=70, color='gray')
    axes[0,1].set_xlabel('number of inter-TAD contacts')
    #axes[0,1].axvline(np.mean(map(rog, Xflat)), ls='--', c='r')
    for spine in ('top', 'left', 'right'):
        axes[0,1].spines[spine].set_visible(False)
    axes[0,1].set_yticks(())


if False:
    fig, axes = plt.subplots(2,2)

    fig.tight_layout()
    plt.show()

if not True:
    fig, ax = plt.subplots()
    plot_TADcm_hist(ax)

    path = os.path.expanduser('~/projects/ehic-paper/nmeth/supplementary_information/figures/nora_TADcmdistance_histograms/')
    fig.savefig(path + '{}structures{}.svg'.format(n_structures, rep))
    fig.savefig(path + '{}structures{}.pdf'.format(n_structures, rep))
    
