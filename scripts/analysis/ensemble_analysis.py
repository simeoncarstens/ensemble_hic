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
    # config_file = '/scratch/scarste/ensemble_hic/bau5C_test/config.cfg'
    settings = parse_config_file(config_file)
    n_replicas = 40
    target_replica = n_replicas
    burnin = 5000

    output_folder = settings['general']['output_folder']
    if output_folder[-1] != '/':
        output_folder += '/'
    n_structures = int(settings['general']['n_structures'])

    samples = load_sr_samples(output_folder + 'samples/', n_replicas, 20001, 100,
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
# cmethod = 'kmeans'

if True:
    ## cluster all structures
    n_clusters_range = range(2,20)
    all_labels = []
    all_silhouette_scores = []
    if cmethod == 'kmeans':
        dms = np.array([pdist(x) for X in ens for x in X])
        for n_clusters in n_clusters_range:
            print "Running kmeans for k={} clusters...".format(n_clusters)
            _, labels = kmeans2(dms, n_clusters, 10, check_finite=False)
            all_labels.append(labels)
            all_silhouette_scores.append(silhouette_score(dms, labels))
    if cmethod == 'spectral':
        a = ens[::3]
        ens_flat = a.reshape(a.shape[0] * a.shape[1], -1, 3)
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
    all_best_index = np.argmax(all_silhouette_scores)
    all_best_index = n_clusters_range.index(13)
    all_best_labels = all_labels[all_best_index]
    n_counts = [(k,np.sum(all_best_labels==k)) for k in range(max(all_best_labels)+1)]
    n_counts.sort(lambda a, b: cmp(a[1],b[1]))
    n_counts.reverse()
    mapping  = {b[0]: a for a, b in enumerate(n_counts)}

    new_labels = np.array(map(mapping.__getitem__, all_best_labels))
    all_best_labels = new_labels
    if cmethod == 'kmeans':
        all_ddms = np.array([[np.linalg.norm(dms[i] - dms[j])
                              for j in range(len(dms))]
                             for i in range(len(dms))])
    if cmethod == 'spectral':
        all_ddms = rmsds
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.bar(n_clusters_range, all_silhouette_scores)
    ax.set_xticks(n_clusters_range)
    ax.set_xlabel('# of clusters')
    ax.set_ylabel('silhouette score')

    indices = np.argsort(new_labels)
    
    ax = fig.add_subplot(122)
    ## perm = np.argsort(all_best_labels)
    ## sorted_all_ddms = all_ddms[perm].T
    ## sorted_all_ddms = sorted_all_ddms[perm].T
    ## ax.matshow(sorted_all_ddms)
    ax.matshow(np.take(np.take(all_ddms, indices, 1), indices, 0), origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


if not True:
    ## cluster slots separately
    n_clusters_range = range(2,8)
    all_labels = []
    silhouette_scores = []
    all_dms = []
    for n_clusters in n_clusters_range:
        all_labels.append([])
        silhouette_scores.append([])
        all_dms.append([])
        for i in range(n_structures):
            dms = np.array([pdist(X[i]) for X in ens])
            _, labels = kmeans2(dms, n_clusters, 10, check_finite=False, minit='points')
            all_labels[-1].append(labels)
            silhouette_scores[-1].append(silhouette_score(dms, labels))
            all_dms[-1].append(dms)
    all_labels = np.array(all_labels)
    all_dms = np.array(all_dms)
    silhouette_scores = np.array(silhouette_scores)

    best_index = np.argmax(silhouette_scores.mean(1))
    labels = all_labels[best_index]
    n_clusters = n_clusters_range[best_index]
    n_cluster_members = np.array([[sum(labels[i] == k) for k in range(n_clusters)]
                                     for i in range(n_structures)])
    dms = all_dms[best_index]

    fig = plt.figure()
    for slot in range(n_structures):
        ax = fig.add_subplot(4, 4, slot + 1)
        H = np.histogram(labels[slot], bins=range(n_clusters+1))
        ax.bar(range(n_clusters), H[0])
        ax.set_xlabel('cluster label')
        ax.set_xticks(range(n_clusters))
        ax.set_title('slot #{}'.format(slot))
    fig.tight_layout()
    plt.show()


    fig = plt.figure()
    for slot in range(n_structures):
        ax = fig.add_subplot(4, 4, slot + 1)
        current_dms = dms[slot]
        ddms = np.array([[np.linalg.norm(current_dms[j] - current_dms[i]) for j in range(len(current_dms))]
                         for i in range(len(current_dms))])
        current_labels = labels[slot]
        perm = np.argsort(current_labels)
        sorted_ddms = ddms[perm].T
        sorted_ddms = sorted_ddms[perm].T
        ax.matshow(sorted_ddms)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('slot #{}'.format(slot))    
    
    fig.tight_layout()
    plt.show()
