import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering
from csb.bio.utils import rmsd, distance_matrix, radius_of_gyration
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform

from ensemble_hic.setup_functions import parse_config_file
from ensemble_hic.analysis_functions import load_sr_samples

if True:
    config_file = '/scratch/scarste/ensemble_hic/proteins/1pga_1shf_maxwell_poisson_es100_sigma0.05_radius0.5_4structures_s_40replicas/config.cfg'
    config_file = sys.argv[1]
    settings = parse_config_file(config_file)
    n_replicas = 80
    target_replica = n_replicas
    n_samples = settings['replica']['n_samples']
    dump_interval = settings['replica']['dump_interval']
    burnin = 10000

    output_folder = settings['general']['output_folder']
    if output_folder[-1] != '/':
        output_folder += '/'
    n_structures = int(settings['general']['n_structures'])

    samples = load_sr_samples(output_folder + 'samples/', n_replicas, n_samples, dump_interval,
                              burnin=burnin)
    samples = samples[None,:]
    if 'weights' in samples[-1,-1].variables:
        weights = np.array([x.variables['weights'] for x in samples.ravel()])
    if 'norm' in samples[-1,-1].variables:
        norms = np.array([x.variables['norm'] for x in samples.ravel()])

    ens = np.array([sample.variables['structures'].reshape(n_structures, -1, 3) for sample in  samples[-1,::5]])
else:
    ens = np.load('ensemble.npy')

cmethod = 'spectral'
save_figures = False
figures_folder = output_folder + 'analysis/clustering/'
if not os.path.exists(output_folder + 'analysis'):
    os.makedirs(directory)
if not os.path.exists(figures_folder):
    os.makedirs(directory)
    

def spectral_clustering_RMSDs(ens, n_clusters_range):

    all_labels = []
    all_silhouette_scores = []
    
    ens_flat = ens.reshape(ens.shape[0] * ens.shape[1], -1, 3)
    rmsds = squareform([rmsd(ens_flat[i], ens_flat[j])
                        for i in range(len(ens_flat))
                        for j in range(i+1, len(ens_flat))])
    for n_clusters in n_clusters_range:
        print "Running spectral clustering for k={} clusters...".format(n_clusters)
        labels = spectral_clustering(np.exp(-rmsds/10.0), n_clusters=n_clusters,
                                     eigen_solver='arpack')
        all_labels.append(labels)
        all_silhouette_scores.append(silhouette_score(np.exp(-rmsds/10.0),
                                                      labels))
    best_index = np.argmax(all_silhouette_scores)
    best_labels = all_labels[all_best_index]
    n_counts = [(k,np.sum(best_labels==k)) for k in range(max(best_labels)+1)]
    n_counts.sort(lambda a, b: cmp(a[1],b[1]))
    n_counts.reverse()
    mapping  = {b[0]: a for a, b in enumerate(n_counts)}
    best_labels = np.array(map(mapping.__getitem__, best_labels))

    return all_silhouette_scores, best_labels, rmsds
    

def kmeans_clustering_distances(ens, n_clusters_range):
    raise NotImplementedError
    all_labels = []
    all_silhouette_scores = []
    dms = np.array([pdist(x) for X in ens for x in X])
    for n_clusters in n_clusters_range:
        print "Running kmeans for k={} clusters...".format(n_clusters)
        _, labels = kmeans2(dms, n_clusters, 50, check_finite=False)
        all_labels.append(labels)
        all_silhouette_scores.append(silhouette_score(dms, labels))

    best_index = np.argmax(all_silhouette_scores)
    best_labels = all_labels[best_index]
    n_counts = [(k,np.sum(best_labels==k)) for k in range(max(best_labels)+1)]
    n_counts.sort(lambda a, b: cmp(a[1],b[1]))
    n_counts.reverse()
    mapping  = {b[0]: a for a, b in enumerate(n_counts)}
    best_labels = np.array(map(mapping.__getitem__, all_best_labels))

    all_ddms = np.array([[np.linalg.norm(dms[i] - dms[j])
                          for j in range(len(dms))]
                          for i in range(len(dms))])
    
    return all_silhouette_scores, best_labels, all_ddms
    


if True:
    ## cluster all structures
    n_clusters_range = range(2, 20)
    if cmethod == 'spectral':
        sh_scores, best_labels, dmatrix = spectral_clustering_RMSDs(ens, n_clusters_range)
    if cmethod == 'kmeans':
        sh_scores, best_labels, dmatrix = kmeans_clustering_RMSDs(ens, n_clusters_range)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.bar(n_clusters_range, sh_scores)
    ax.set_xticks(n_clusters_range)
    ax.set_xlabel('# of clusters')
    ax.set_ylabel('silhouette score')

    ax = fig.add_subplot(122)
    indices = np.argsort(best_labels)
    sorted_dmatrix = np.take(np.take(dmatrix, indices, 1), indices, 0)
    ax.matshow(sorted_dmatrix, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])

    if save_figures:
        plt.savefig(figures_folder + 'all_structures.pdf')
    else:
        plt.show()


if True:
    ## cluster slots separately
    n_clusters_range = range(2,20)
    results = []
    for i in range(n_structures):
        subens = ens[:,i]
        if cmethod == 'spectral':
            sh_scores, best_labels, dmatrix = spectral_clustering_RMSDs(subens, n_clusters_range)
        if cmethod == 'kmeans':
            sh_scores, best_labels, dmatrix = kmeans_clustering_RMSDs(subens, n_clusters_range)
        results.append([sh_scores, best_labels, dmatrix])

    fig = plt.figure()
    for slot in range(n_structures):
        ax = fig.add_subplot(4, 5, slot + 1)
        n_clusters = max(results[slot][1])
        H = np.histogram(best_labels[slot], bins=range(n_clusters))
        ax.bar(range(n_clusters), H[0])
        ax.set_xlabel('cluster label')
        ax.set_xticks(range(n_clusters))
        ax.set_title('slot #{}'.format(slot))
    fig.tight_layout()
    if save_figures:
        plt.savefig(figures_folder + 'single_slots_clustersizes.pdf')
    else:
        plt.show()

    fig = plt.figure()
    for slot in range(n_structures):
        ax = fig.add_subplot(4, 5, slot + 1)
        dmatrix = results[slot][2]
        indices = np.argsort(results[slot][1])
        sorted_dmatrix = np.take(np.take(dmatrix, indices, 1), indices, 0) 
        ax.matshow(sorted_ddms)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('slot #{}'.format(slot))    
    
    fig.tight_layout()
    if save_figures:
        plt.savefig(figures_folder + 'single_slots_dmatrices.pdf')
    else:
        plt.show()

    fig = plt.figure()
    for slot in range(n_structures):
        ax = fig.add_subplot(4, 5, slot + 1)
        ax.bar(n_clusters_range, results[slot][0])
        ax.set_xticks(n_clusters_range)
        ax.set_xlabel('# of clusters')
        ax.set_ylabel('silhouette score')
        ax.set_title('slot #{}'.format(slot))
        
    fig.tight_layout()
    if save_figures:
        plt.savefig(figures_folder + 'single_slots_shscores.pdf')
    else:
        plt.show()
