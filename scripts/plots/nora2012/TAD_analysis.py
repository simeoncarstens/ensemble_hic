import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import rmsd, radius_of_gyration as rog

from ensemble_hic.analysis_functions import load_sr_samples

n_structures = 40
n_replicas = 252
n_samples = 36000
burnin = 20000
it = ''
rep = ''
data = 'female_'

n_structures = 40
n_replicas = 252
n_samples = 9001
burnin = 4000
it = ''
rep = ''
data = 'female_day2_'

n_structures = 20
n_replicas = 217
n_samples = 16001
burnin = 10000
it = ''
rep = ''
data = 'female_day2_'

n_structures = 20
n_replicas = 217
n_samples = 12001
burnin = 7000
it = ''
rep = ''
data = 'female_'

# n_structures = 40
# n_replicas = 330
# n_samples = 33000
# burnin = 20000
# it = 2

simpath = '/scratch/scarste/ensemble_hic/nora2012/{}bothdomains{}{}_{}structures_{}replicas/'.format(data, it, rep, n_structures, n_replicas)

s = load_sr_samples(simpath + 'samples/', n_replicas, n_samples, 1000, burnin)
X = np.array([x.variables['structures'].reshape(n_structures, 308, 3)
              for x in s]) * 53
Xflat = X.reshape(-1,308,3)

pos_start = 100378306

t1 = X[:,:,:107]
t2 = X[:,:,107:]
t1flat = t1.reshape(-1, 107,3)
t2flat = t2.reshape(-1, 201,3)

fig, axes = plt.subplots(2,2)

## gyration radius histograms
rogs_t1 = np.array(map(rog, t1flat))
rogs_t2 = np.array(map(rog, t2flat))
axes[0,0].hist(rogs_t1, bins=100, label='Tsix TAD', alpha=0.6)
axes[0,0].hist(rogs_t2, bins=100, label='Xist TAD', alpha=0.6)
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
    ds = np.array([np.linalg.norm(t1flat[i].mean(0) - t2flat[i].mean(0))
                   for i in range(len(Xflat))])
    axes[0,1].hist(ds, bins=70)
    axes[0,1].set_xlabel('TAD center of mass distances [nm]')
    axes[0,1].axvline(np.mean(map(rog, Xflat)), ls='--', c='r')
    for spine in ('top', 'left', 'right'):
        axes[0,1].spines[spine].set_visible(False)
    axes[0,1].set_yticks(())

fig.tight_layout()
plt.show()
