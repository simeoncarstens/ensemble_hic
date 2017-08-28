import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering
from csb.bio.utils import rmsd, distance_matrix, radius_of_gyration
from csb.bio.io.wwpdb import StructureParser
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform

from ensemble_hic.analysis_functions import load_sr_samples

data_dir = os.path.expanduser('~/projects/ensemble_hic/data/proteins/')

n_structures = 1
variables = 's'
n_replicas = 40
burnin = 5000

fname1 = '1pga'
fname2 = '1shf'

# fname1 = '1ubq'
# fname2 = '2ma1'

known1 = StructureParser(data_dir + fname1 + '.pdb').parse().get_coordinates(['CA'])
known2 = StructureParser(data_dir + fname2 + '.pdb').parse().get_coordinates(['CA'])

known1 /= 3.8
known2 /= 3.8

results_path = '/scratch/scarste/ensemble_hic/proteins/'
results_path = '{}{}_{}_poisson_radius0.5_{}structures_{}_{}replicas/'.format(results_path, fname1, fname2,  n_structures, variables, n_replicas)

# results_path = '/scratch/scarste/ensemble_hic/protein_isn1_wzeros_cd1.8ss14_kbb1000_mpdata_es100_sigma0.05_{}_{}_poisson_{}_{}structures_40replicas/'.format(fname1, fname2, variables, n_structures)
results_path = '/scratch/scarste/ensemble_hic/protein_isn1_wzeros_cd1.8ss50_kbb1000_pdata_{}_{}_poisson_{}_{}structures_40replicas/'.format(fname2,
                                                                                                                                            'none',#fname2,
                                                                                                                                            variables, n_structures)


samples = load_sr_samples(results_path + 'samples/', n_replicas, 20001, 100,
                          burnin=burnin)
ens = np.array([sample.variables['structures'].reshape(-1, len(known1), 3)
                for sample in  samples])
ens_flat = ens.reshape(ens.shape[0] * ens.shape[1], -1, 3)

if True:
    ## plot histograms of RMSDs to known structures
    rmsds1 = map(lambda x: rmsd(known1, x), ens_flat)
    rmsds2 = map(lambda x: rmsd(known2, x), ens_flat)
    max_rmsd = max(rmsds1 + rmsds2)
    min_rmsd = min(rmsds1 + rmsds2)
    

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.hist(rmsds1, bins=len(ens_flat))
    ax.set_xlabel('RMSD to ' + fname1)
    ax.set_xlim((min_rmsd, max_rmsd))
    
    ax = fig.add_subplot(212)
    ax.hist(rmsds2, bins=len(ens_flat))
    ax.set_xlabel('RMSD to ' + fname2)
    ax.set_xlim((min_rmsd, max(rmsds1 + rmsds2)))

    fig.tight_layout()
    plt.show()
