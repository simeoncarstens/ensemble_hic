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

samples = load_sr_samples(sim_path + 'samples/', n_replicas, n_samples,
                          dump_interval, burnin)
X = np.array([x.variables['structures'].reshape(-1, n_beads, 3) for x in samples])
Xflat = X.reshape(-1, n_beads, 3)

samples_null = load_sr_samples(null_path + 'samples/', 100, 23301, 100, 10000, 5)
Xnull = np.array([x.variables['structures'].reshape(-1, n_beads, 3)
                  for x in samples_null])
Xflatnull = Xnull.reshape(-1, n_beads, 3)


dms = np.array([pdist(x) for x in Xflat[np.random.choice(len(Xflat), n_dms)]])
dmsnull = np.array([pdist(x) for x in Xflatnull[np.random.choice(len(Xflatnull),
                                                                 n_dms)]])


#dms = dms[:,:10]
#dmsnull = dmsnull[:,:10]

from ensemble_hic.analysis_functions import calculate_KL_KDE_log as calculate_KL_KDE
from multiprocessing import Pool
pool = Pool(16)
from time import time
a=time()
KL = pool.map(calculate_KL_KDE, ((dms[:,i], dmsnull[:,i], ) for i in range(dms.shape[1])))
print time() - a
np.save('/usr/users/scarste/test/KL_yeast_50_KDE_log.npy', KL)
