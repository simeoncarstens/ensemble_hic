import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import rmsd, radius_of_gyration as rog, distance_matrix

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

cgen_ss = lambda d, a, cutoff, offset: np.triu((a*(cutoff-d)/np.sqrt(1+a*a*(d-cutoff)*(d-cutoff))+1)*0.5, offset)

def find_TADs(x, cutoff=1.5, offset=(3,10)[1]):
    a = 10.0
    d = distance_matrix(x)
    c = cgen_ss(d, a, cutoff, offset)
    j = np.arange(len(x))
    counts = np.array([c[:i,:i].sum() + c[i:,i:].sum() for i in j])
    areas  = j**2 + (len(x) - j)**2

    return np.argmax(counts.astype('d') / areas)

def find_TADs_pop(X, cutoff=1.5, offset=(3,10)[1]):
    a = 10.0
    d = np.array([squareform(pdist(x)) for x in X])
    c = np.sum(map(lambda sd: cgen_ss(sd, a, cutoff, offset), d), axis=0) / len(d)
    j = np.arange(len(X[0]))
    counts = np.array([c[:i,:i].sum() + c[i:,i:].sum() for i in j])
    areas  = j**2 + (len(X[0]) - j)**2

    return np.argmax(counts.astype('d') / areas)

def plot_TAD_boundary_hists(ax):
    
    cutoff = 2.0
    offset = 6
    n_samples = 1000

    np.random.seed(32)
    random = lambda n: np.random.choice(np.arange(len(X)), n)
    if False:
        scores_pop = np.array(map(lambda x: find_TADs_pop(x, cutoff, offset),
                                  X[random(n_samples)] / 53.0))
        np.save(os.path.expanduser('~/test/scores_pop.npy'), scores_pop)
    else:
        scores_pop = np.load(os.path.expanduser('~/test/scores_pop.npy'))
    if False:
        rinds = random(n_samples * 10)
        scores = np.array(map(lambda x: find_TADs(x, cutoff, offset),
                              X.reshape(-1,308,3)[rinds] / 53.0))
        np.save(os.path.expanduser('~/test/scores.npy'), scores)
        np.save(os.path.expanduser('~/test/rinds.npy'), rinds)
    else:
        scores = np.load(os.path.expanduser('~/test/scores.npy'))
        rinds = np.load(os.path.expanduser('~/test/rinds.npy'))
    hargs = dict(alpha=0.5, histtype='stepfilled', normed=True)
    ax.hist(scores, label='single structures', color='gray',
                 bins=np.arange(0,308,2), **hargs)
    ax.hist(scores_pop, label='structure populations', color='black',
            bins=np.arange(0,308,1), **hargs)
    print scores.mean(), scores_pop.mean()
    print scores.min(), scores.max()
    ax.set_xlabel('TAD boundary position [beads]')
    ax.set_xlim(50, 225)
    ax.yaxis.set_visible(False)
    for spine in ('top', 'left', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    ax.set_yscale('log')

    from ensemble_hic.setup_functions import make_posterior, parse_config_file
    settings = parse_config_file(sim_path + 'config.cfg')
    settings['general']['n_structures'] = '1'
    p = make_posterior(settings)
    fwm = p.likelihoods['ensemble_contacts'].forward_model
    data = fwm.data_points
    xmin = X.reshape(-1, 308, 3)[rinds[np.argmin(scores)]]
    xmax = X.reshape(-1, 308, 3)[rinds[np.argmax(scores)]]
    np.save('./xmin.npy', xmin)
    np.save('./xmax.npy', xmax)

    fig2, (ax2, ax3) = plt.subplots(1,2)

    if not True:
        def bla(x):
            s = lambda d: 0.5*(10.0 * d/np.sqrt(1.0 + 10 * 10 * d * d) + 1.0)
            from csb.bio.utils import distance_matrix
            ds = distance_matrix(x)
            br = p.priors['nonbonded_prior'].forcefield.bead_radii
            mus = s(np.add.outer(br, br) * 1.25 * 53 - ds)
            mus2 = np.zeros((308,308))
            mus2[data[:,0], data[:,1]] = mus[data[:,0], data[:,1]]
            mus2[data[:,1], data[:,0]] = mus[data[:,1], data[:,0]]
            return ds#mus2

        m2 = bla(xmin)
        m3 = bla(xmax)

        ax2.axvline(71, ls='--', c='r')
        ax2.axhline(71, ls='--', c='r')
        ax3.axvline(203, ls='--', c='r')
        ax3.axhline(203, ls='--', c='r')

        ax2.matshow(m2)
        ax3.matshow(m3)

    else:
        md_min = fwm(structures=xmin.ravel(), norm=1.0)
        md_max = fwm(structures=xmax.ravel(), norm=1.0)
    
        m2 = np.zeros((308,308))
        m3 = np.zeros((308,308))
        m2[data[:,0], data[:,1]] = md_min
        m2[data[:,1], data[:,0]] = md_min
        m3[data[:,1], data[:,0]] = md_max
        m3[data[:,0], data[:,1]] = md_max
        inds = np.arange(308)
        nonzero_inds = np.array([i for i in inds if m2[i,:].sum() > 0])
        m2 = np.log(m2[nonzero_inds,:][:,nonzero_inds] + 1e-3)
        m3 = np.log(m3[nonzero_inds,:][:,nonzero_inds] + 1e-3)
        
        ax2.matshow(m2)
        ax3.matshow(m3)

        for acks in (ax2, ax3):
            
            step = 15
            ticks = np.arange(0, len(nonzero_inds))[::step]
            labels = map(str, nonzero_inds)[::step]
            acks.set_xticks(ticks)
            acks.set_xticklabels(labels)
            acks.set_yticks(ticks)
            acks.set_yticklabels(labels)
