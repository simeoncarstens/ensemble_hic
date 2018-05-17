import numpy as np
from scipy.spatial.distance import squareform, pdist, hamming
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering

from csb.bio.utils import rmsd, wrmsd

def contact_hamming_affinities(ens, contact_distances,
                               weights, n_processes=10):
    
    from multiprocessing import Pool
    pool = Pool(n_processes)
    contacts = np.array(map(pdist, ens)) < contact_distances[None,:]
    rmsds = pool.map(hamming_pair, [(contacts[i], contacts[j], weights)
                                    for i in range(len(ens))
                                    for j in range(i+1, len(ens))])
    pool.close()
    rmsds = squareform(np.array(rmsds))

    return rmsds

def hamming_pair((contacts1, contacts2, weights)):

    ## not in my SciPy version
    # return hamming(contacts1, contacts2, weights)
    
    return hamming(contacts1, contacts2)

def rmsd_affinities(ens, n_processes=10):
    from multiprocessing import Pool
    pool = Pool(n_processes)
    rmsds = pool.map(rmsd_pair, [(ens[i], ens[j])
                                 for i in range(len(ens))
                                 for j in range(i+1, len(ens))])
    pool.close()
    rmsds = squareform(np.array(rmsds))

    return rmsds#np.exp(-rmsds / rmsds.max())

def rmsd_pair((x1, x2)):

    return rmsd(x1, x2)

def wrmsd_affinities(ens, masses, n_processes=10):

    from multiprocessing import Pool
    pool = Pool(n_processes)
    rmsds = pool.map(wrmsd_pair, [(ens[i], ens[j], masses)
                                  for i in range(len(ens))
                                  for j in range(i+1, len(ens))])
    pool.close()
    rmsds = squareform(np.array(rmsds))

    return rmsds#np.exp(-rmsds / rmsds.max())

def wrmsd_pair((x1, x2, masses)):

    return wrmsd(x1, x2, masses)
    
def distance_rmsd_affinities(ens, n_processes=10):
    from multiprocessing import Pool
    pool = Pool(n_processes)
    dms = map(lambda x: squareform(pdist(x)), ens)
    rmsds = pool.map(distance_rmsd_pair, [(dms[i], dms[j])
                                       for i in range(len(ens))
                                       for j in range(i+1, len(ens))])
    pool.close()
    rmsds = squareform(np.array(rmsds))
    
    return rmsds#np.exp(-rmsds / rmsds.max())

def distance_rmsd_pair((dm1, dm2)):

    return np.sum((dm1 - dm2) ** 2)

def weighted_distance_rmsd_pair((x,y,brsum)):

    return np.sum(brsum * (pdist(x) - pdist(y)) ** 2)

def weighted_distance_rmsd_affinities(ens, bead_radii):
    brsum = np.add.outer(bead_radii, bead_radii) ** 2
    brsum[np.diag_indices(len(brsum))] = 0.0
    brsum = squareform(brsum)
    brsum /= sum(brsum)

    from multiprocessing import Pool
    pool = Pool(10)
    rmsds = pool.map(weighted_distance_rmsd_pair, [(ens[i], ens[j], brsum)
                                                   for i in range(len(ens))
                                                   for j in range(i+1, len(ens))])
    pool.close()
    rmsds = squareform(np.array(rmsds))
    
    return np.exp(-rmsds / rmsds.max())

def coarse_fwm(x, cds, dps, alpha):

    from ensemble_hic.forward_models import EnsembleContactsFWM
    FWM = EnsembleContactsFWM('asdf', 1, cds, dps, cutoff=5000.0)
    w=np.ones(1)
    bla = FWM(structures=x.ravel(), weights=w, norm=1, 
              smooth_steepness=alpha)
    m = np.zeros((70, 70))
    m[dps[:,0], dps[:,1]] = bla
    bl = 7

    return np.array([m[i*bl:(i+1)*bl,j*bl:(j+1)*bl].sum()
                     for i in range(len(m)/bl)
                     for j in range(i,len(m)/bl) if j-i>1])

def coarse_fwm_pair((x, y, cds, dps, alpha)):
    return np.sum((coarse_fwm(x, cds, dps, alpha) - coarse_fwm(y, cds, dps, alpha))**2)
    
def coarse_fwm_affinities(ens, cds, dps, alpha, n_processes=10):

    from multiprocessing import Pool
    pool = Pool(10)
    rmsds = pool.map(coarse_fwm_pair, [(ens[i], ens[j], cds, dps, alpha)
                                       for i in range(len(ens))
                                       for j in range(i+1, len(ens))])
    pool.close()
    rmsds = squareform(np.array(rmsds))
    
    return np.exp(-rmsds / rmsds.max())

def linear_density(x, bead_radii):

    cutoff = 2.0

    dm = squareform(pdist(x))
    res = [np.sum((dm[i] < cutoff) * (bead_radii[i] + bead_radii))
           for i in range(len(dm))]

    return np.array([np.sum(res[i*7:(i+1)*7]) for i in range(9)])

def linear_density_pair((x, y, bead_radii)):

    return np.sum((linear_density(x, bead_radii) - linear_density(y, bead_radii))**2)

def linear_density_affinities(ens, bead_radii, n_processes=10):

    from multiprocessing import Pool
    pool = Pool(n_processes)
    rmsds = pool.map(linear_density_pair, [(ens[i], ens[j], bead_radii)
                                           for i in range(len(ens))
                                           for j in range(i+1, len(ens))])
    pool.close()
    rmsds = squareform(np.array(rmsds))

    return np.exp(-rmsds / rmsds.max())

def get_labels(rank, all_labels, all_silhouette_scores):

    index = np.argsort(all_silhouette_scores)[::-1][rank]
    labels = all_labels[index]
    n_counts = [(k,np.sum(labels==k)) for k in range(max(labels)+1)]
    n_counts.sort(lambda a, b: cmp(a[1],b[1]))
    n_counts.reverse()
    mapping  = {b[0]: a for a, b in enumerate(n_counts)}
    labels = np.array(map(mapping.__getitem__, labels))

    return labels    

def perform_clustering(n_clusters_range, affinities):

    all_labels = []
    all_silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        labels = spectral_clustering(affinities, n_clusters=n_clusters,
                                     eigen_solver='arpack')
        all_labels.append(labels)
        all_silhouette_scores.append(silhouette_score(affinities,
                                                      labels))

    return all_silhouette_scores, all_labels, affinities
