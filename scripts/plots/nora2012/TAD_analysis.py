import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import rmsd, radius_of_gyration as rog

from ensemble_hic.analysis_functions import load_sr_samples

n_beads = 308

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_fixed_it3_rep3_20structures_309replicas/'
s = load_sr_samples(sim_path + 'samples/', 309, 50001, 1000, 30000)
X = np.array([x.variables['structures'].reshape(20, 308, 3)
              for x in s]) * 53

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_nointer_it3_rep3_20structures_309replicas/'
s = load_sr_samples(sim_path + 'samples/', 309, 50001, 1000, 30000)
X_nointer = np.array([x.variables['structures'].reshape(20, 308, 3)
                      for x in s]) * 53

pos_start = 100378306


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
    def plot_TADcm_hist(ax, X, label, color):
        Xflat = X.reshape(-1,308,3)
        t1 = X[:,:,:107]
        t2 = X[:,:,107:]
        t1flat = t1.reshape(-1, 107,3)
        t2flat = t2.reshape(-1, 201,3)
        ds = np.array([np.linalg.norm(t1flat[i].mean(0) - t2flat[i].mean(0))
                       for i in range(len(Xflat))])
        ax.hist(ds, bins=50, color=color, histtype='stepfilled', alpha=0.6,
                label=label)
        ax.set_xlim(0, 850)
        ax.set_xlabel('TAD center of mass distances [nm]')
        ax.axvline(np.mean(map(rog, Xflat)), ls='--', c=color)
        for spine in ('top', 'left', 'right'):
            ax.spines[spine].set_visible(False)
        ax.set_yticks(())

    def plot_TADiness(ax, X, label, color):
        Xflat = X.reshape(-1,308,3)
        t1 = X[:,:,:107]
        t2 = X[:,:,107:]
        t1flat = t1.reshape(-1, 107,3)
        t2flat = t2.reshape(-1, 201,3)

        def find_engulfing_sphere_radius(t):
            cm = t.mean(0)
            d_to_cm = np.linalg.norm(t-cm, axis=1)
            return max(d_to_cm)

        print find_engulfing_sphere_radius(t1)

t1 = X[:,:,:107]
t2 = X[:,:,107:]
t1flat = t1.reshape(-1, 107,3)
t2flat = t2.reshape(-1, 201,3)

def find_engulfing_sphere_radius(x):
    if True:
        cm = x.mean(0)
        d_to_cm = np.linalg.norm(x - cm, axis=0)
        return max(d_to_cm) / float(len(x) ** 3)
    if False:
        from csb.bio.utils import radius_of_gyration
        return radius_of_gyration(x) / float(len(x) ** 3)

if False:
    print find_engulfing_sphere_radius(t1flat[0])
    a = lambda x: np.array(map(lambda i: find_engulfing_sphere_radius(x[:i]), range(1,len(x))))
    bla = np.array(map(a, X_nointer.reshape(-1,308,3)[-1000:]))

if True:
    from ensemble_hic.setup_functions import make_posterior, parse_config_file
    settings = parse_config_file(sim_path + 'config.cfg')
    settings['general']['n_structures'] = '1'
    p = make_posterior(settings)
    fwm = p.likelihoods['ensemble_contacts'].forward_model
    contacts = fwm.data_points[:,:2]

    def find_TADs(x):
        md = fwm(structures=x[None,:], norm=1.0)
        m = np.zeros((308,308))
        m[contacts[:,0], contacts[:,1]] = md
        m[contacts[:,1], contacts[:,0]] = md
        ncts = []
        scores = [m[:i,:i].sum() + m[i:,i:].sum() - 0.0001*(i ** 2 + (308 - i) ** 2)
                  for i in range(308)]
        return np.argmax(scores)

    
    from csb.bio.utils import distance_matrix
    cgen_ss = lambda d, a, cutoff, offset: np.triu((a*(cutoff-d)/np.sqrt(1+a*a*(d-cutoff)*(d-cutoff))+1)*0.5, offset)

    def find_TADs(x, cutoff=1.5, offset=(3,10)[1]):
        d = distance_matrix(x)
        a = p['smooth_steepness'].value
        c = cgen_ss(d, a, cutoff, offset)
        j = np.arange(len(x))
        counts = np.array([c[:i,:i].sum() + c[i:,i:].sum() for i in j])
        areas  = j**2 + (len(x) - j)**2
        
        return np.argmax(counts.astype('d') / areas)

    def find_TADs_pop(X, cutoff=1.5, offset=(3,10)[1]):
        d = np.array([squareform(pdist(x)) for x in X])
        a = p['smooth_steepness'].value
        c = np.sum(map(lambda sd: cgen_ss(sd, a, cutoff, offset), d), axis=0) / len(d)
        j = np.arange(len(X[0]))
        counts = np.array([c[:i,:i].sum() + c[i:,i:].sum() for i in j])
        areas  = j**2 + (len(X[0]) - j)**2
        
        return np.argmax(counts.astype('d') / areas)

    cutoff = 2.0
    offset = 10

    random = lambda n: np.random.choice(np.arange(len(X)), n)
    scores = np.array(map(lambda x: find_TADs(x, cutoff, offset),
                          X.reshape(-1,308,3)[random(1000)] / 53.0))
    scores_nointer = np.array(map(lambda x: find_TADs(x, cutoff, offset),
                                  X_nointer.reshape(-1,308,3)[random(1000)] / 53.0))
    scores_pop = np.array(map(lambda x: find_TADs_pop(x, cutoff, offset),
                              X[random(1000)] / 53.0))
    scores_nointer_pop = np.array(map(lambda x: find_TADs_pop(x, cutoff, offset),
                                      X_nointer[random(1000)] / 53.0))

    fig, (ax1, ax2) = plt.subplots(2,1)
    hargs = dict(alpha=0.6, histtype='stepfilled', normed=True,
                 bins=np.arange(0,307,2))

    ax1.hist(scores, label='full data', **hargs)
    ax1.hist(scores_nointer, label='w/o inter\ncontacts', **hargs)
    ax1.set_xlim(0,308)
    ax1.set_title('single structures')
    ax1.legend()

    ax2.hist(scores_pop, label='full data', **hargs)
    ax2.hist(scores_nointer_pop, label='w/o inter\ncontacts', **hargs)
    ax2.set_xlim(0,308)
    ax2.set_title('structure populations')
    ax2.legend()

    fig.tight_layout()
    plt.show()
    
    
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
    plot_TADcm_hist(ax, X, 'full data', 'gray')
    plot_TADcm_hist(ax, X_nointer, 'no inter-TAD\ncontacts', 'lightgray')
    ax.legend()

    path = os.path.expanduser('~/projects/ehic-paper/nmeth/supplementary_information/figures/nora_TADcmdistance_histograms/')
    fig.savefig(path + '{}structures{}.svg'.format(n_structures, rep))
    fig.savefig(path + '{}structures{}.pdf'.format(n_structures, rep))
    
