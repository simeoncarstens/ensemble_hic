import numpy as np
import os, sys
from scipy.spatial.distance import squareform, pdist

from ensemble_hic.setup_functions import make_posterior, parse_config_file
from ensemble_hic.analysis_functions import load_sr_samples

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/data/eser2017/'))
from yeastlib import CGRep
from model_from_arbona2017 import *

n_structures = 50
n_replicas = 366
suffix = '_it3'
n_samples = 3251
dump_interval = 50
burnin = 1500
n_dms = 400
sim_path = '/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017{}_{}structures_sn_{}replicas/'.format(suffix, n_structures, n_replicas)
null_path = '/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA150_prior2_arbona2017_fixed_1structures_s_100replicas/'

p = make_posterior(parse_config_file(sim_path + 'config.cfg'))
n_beads = p.priors['nonbonded_prior'].forcefield.n_beads

# samples = load_sr_samples(sim_path + 'samples/', n_replicas, n_samples,
#                           dump_interval, burnin)
# X = np.array([x.variables['structures'].reshape(-1, n_beads, 3) for x in samples])
# Xflat = X.reshape(-1, n_beads, 3)

samples_null = load_sr_samples(null_path + 'samples/', 100, 23301, 100, 10000, 5)
Xnull = np.array([x.variables['structures'].reshape(-1, n_beads, 3)
                  for x in samples_null])
Xflatnull = Xnull.reshape(-1, n_beads, 3)


# dms = np.array([pdist(x) for x in Xflat[np.random.choice(len(Xflat), n_dms)]])
dmsnull = np.array([pdist(x) for x in Xflatnull[np.random.choice(len(Xflatnull),
                                                                 n_dms)]])

# from ensemble_hic.analysis_functions import calculate_KL
# from multiprocessing import Pool
# pool = Pool(20)
# bins = np.linspace(0, np.max((dms.max(), dmsnull.max())), int(np.sqrt(n_dms)))
# KL = pool.map(calculate_KL, ((dms[:,i], dmsnull[:,i], bins) for i in range(len(dms[0]))))

# np.save('/usr/users/scarste/test/KL_yeast_50.npy', KL)


if not False:

    from csb.statistics.pdf import Gamma
    from csb.numeric import log

    def calculate_KL_prior((prior_distances, bins)):
        g = Gamma()
        g.estimate(prior_distances)
        prior_hist = np.histogram(prior_distances, bins=bins, normed=True)[0]

        return np.trapz(prior_hist * log(prior_hist / g(bins[:-1])), bins[:-1])


from ensemble_hic.analysis_functions import calculate_KL
from multiprocessing import Pool
pool = Pool(20)
bins = np.linspace(0, dmsnull.max(), int(np.sqrt(n_dms)))
KL = pool.map(calculate_KL_prior, ((dmsnull[:,i], bins) for i in range(len(dmsnull[0]))))

np.save('/usr/users/scarste/test/KL_yeast_50_prior.npy', KL)


if False:
    d = lambda i,j,n: i*n + j - i*(i+1)/2 - i - 1
    i, j = 300, 1005
    sind = d(i, j, 1239)
    maks = max((dms.max(), dmsnull.max()))
    plt.plot([calculate_KL((dms[:,sind], dmsnull[:,sind], np.linspace(0, maks, i)))
              for i in range(20,2000,20)])

    def estimate_shannon(x, bins):

        from csb.numeric import log

        res = np.histogram(x, bins, normed=True)

        return np.trapz(-log(res[0]) * res[0], 0.5 * (res[1][1:] + res[1][:-1]))

    def estimate_shannon2(x, bins):

        from csb.numeric import log

        p, _ = np.histogram(x, bins, normed=True)
        p /= p.sum()
        
        return np.dot(p, -log(p))

    def estimate_cross_entropy(x, x_null, bins, prior=None):

        from csb.numeric import log

        if prior is not None:
            g = prior
        else:
            g = Gamma()
            g.estimate(x_null)

        p, y = np.histogram(x, bins, normed=True)
        y = 0.5 * (y[1:] + y[:-1])
        q = g(y)
        
        return np.trapz(-log(q) * p, y)

    from sklearn.neighbors import KernelDensity

    bins = 100
    figure();
    res1 = hist(dms[:,d(300,805,1239)], bins, alpha=0.6, normed=True)
    res2 = hist(dmsnull[:,d(300,805,1239)], bins, alpha=0.6, normed=True)
    space=linspace(0, dmsnull[:,d(300,805,1239)].max(),100)
    g = Gamma()
    g.estimate(dmsnull[:,d(300,805,1239)])
    plot(space,g(space))

    from csb.numeric import log
    from scipy.integrate import quad

    print np.trapz(-log(res1[0]) * res1[0], 0.5 * (res1[1][1:] + res1[1][:-1]))

    i, j = 300, 805

    x = dms[:,d(i,j,1239)]
    x_null = dmsnull[:,d(i,j,1239)]

    x_max = max(x.max(), x_null.max()) * 1.1
    space=linspace(0, x_max, 100)
    
    h = x.std()*(4./3/len(x))**(1./5) ## silverman

    bw = h * np.linspace(0.1, 10., 100)
    prior = Gamma()
    prior.estimate(x_null)

    KL = []
    
    for h in bw:
    
        posterior = KernelDensity(kernel='gaussian', bandwidth=h).fit(x.reshape(-1,1))

        ce = lambda x: -log(prior(x)) * np.exp(posterior.score(x))
        hh = lambda x: -posterior.score(x) * np.exp(posterior.score(x))

        vals = (quad(ce, 0., x_max)[0], quad(hh, 0., x_max)[0])

        KL.append(vals[0]-vals[1])
        print h, KL[-1]

    figure();
    bins = 30
    hist(x, bins, alpha=0.6, normed=True)
    hist(x_null, bins, alpha=0.6, normed=True)
    plot(space,prior(space))
    plot(space,np.exp(map(posterior.score,space)))

    
    bins = np.linspace(10, 1000, 1000).astype('i')
    CE = [estimate_cross_entropy(x,x_null,b) for b in bins]
    
    H = [estimate_shannon(x,b) for b in bins]
    H2 = [estimate_shannon2(x,b) for b in bins]
