import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering
from csb.bio.utils import rmsd, distance_matrix, radius_of_gyration, average_structure, scale_and_fit
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file
from ensemble_hic.analysis_functions import load_samples_from_cfg, write_ensemble, write_VMD_script, write_pymol_script

step = 10
if True:
    config_file = sys.argv[1]
    config_file = '/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_20structures_sn_112replicas/config.cfg'
    config_file = '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_20structures_sn_122replicas/config.cfg'
    settings = parse_config_file(config_file)
    output_folder = settings['general']['output_folder']
    samples = load_samples_from_cfg(config_file)[::step]
    
    if 'weights' in samples[-1].variables:
        weights = np.array([x.variables['weights'] for x in samples.ravel()])
    if 'norm' in samples[-1].variables:
        norms = np.array([x.variables['norm'] for x in samples.ravel()])
    
    ens = np.array([sample.variables['structures'].reshape(-1, 70, 3)
                    for sample in samples])

    bead_radii = make_posterior(settings).priors['nonbonded_prior'].bead_radii
else:
    ens = np.load('ensemble.npy')

save_figures = True
n_rows = 3
n_cols = 3
figures_folder = output_folder + 'analysis/clustering/'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

def rmsd_affinities(ens):
    rmsds = squareform([rmsd(ens[i], ens[j])
                        for i in range(len(ens))
                        for j in range(i+1, len(ens))])

    return np.exp(-rmsds / 10.0)

def distance_rmsd_affinities(ens):
    rmsds = squareform([np.sum((pdist(ens[i]) - pdist(ens[j])) ** 2)
                        for i in range(len(ens))
                        for j in range(i+1, len(ens))])

    return np.exp(-rmsds / 100000.0)                        

def get_labels(rank, all_labels, all_silhouette_scores):

    index = np.argsort(all_silhouette_scores)[::-1][rank]
    labels = all_labels[index]
    n_counts = [(k,np.sum(labels==k)) for k in range(max(labels)+1)]
    n_counts.sort(lambda a, b: cmp(a[1],b[1]))
    n_counts.reverse()
    mapping  = {b[0]: a for a, b in enumerate(n_counts)}
    labels = np.array(map(mapping.__getitem__, labels))

    return labels    

def perform_clustering(ens, n_clusters_range, affinities):

    all_labels = []
    all_silhouette_scores = []
    affinities = affinities(ens)
    
    for n_clusters in n_clusters_range:
        labels = spectral_clustering(affinities, n_clusters=n_clusters,
                                     eigen_solver='arpack')
        all_labels.append(labels)
        all_silhouette_scores.append(silhouette_score(affinities,
                                                      labels))

    return all_silhouette_scores, all_labels, affinities


if True:
    ## cluster all structures
    n_clusters_range = range(2, 10)
    affinities = distance_rmsd_affinities
    ens_flatter = ens.reshape(ens.shape[0] * ens.shape[1], -1, 3)
    res = perform_clustering(ens_flatter,
                             n_clusters_range, affinities)
    sh_scores, all_labels, dmatrix = res
    n = min(n_clusters_range)
    while n in n_clusters_range:
        rank = n - min(n_clusters_range)
        labels = get_labels(rank, all_labels, sh_scores)
        outf = figures_folder + 'rank{}_clustering/'.format(rank)
        if not os.path.exists(outf):
            os.makedirs(outf)
        for k in xrange(n):
            members = ens_flatter[labels == k]
            avg_X = average_structure(members)
            Rt = [scale_and_fit(avg_X, Y)[:2] for Y in members]
            aligned_ens = np.array([np.dot(members[i], R.T) + t for i, (R, t)
                                    in enumerate(Rt)])
            write_ensemble(aligned_ens, outf + 'cluster{}.pdb'.format(k))
            write_VMD_script(outf + 'cluster{}.pdb'.format(k), bead_radii,
                             outf + 'cluster{}.rc'.format(k))
            write_pymol_script(outf + 'cluster{}.pdb'.format(k), bead_radii,
                             outf + 'cluster{}.pml'.format(k))
        n += 1

        
        
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


if not True:
    ## cluster slots separately
    n_clusters_range = range(2,10)
    results = []
    for i in range(n_structures):
        subens = ens[:,i]
        if cmethod == 'spectral':
            sh_scores, best_labels, dmatrix = spectral_clustering_RMSDs(subens, n_clusters_range)
        results.append([sh_scores, best_labels, dmatrix])

    fig = plt.figure()
    for slot in range(n_structures):
        ax = fig.add_subplot(n_rows, n_cols, slot + 1)
        n_clusters = max(results[slot][1])
        H = np.histogram(results[slot][1], bins=range(n_clusters + 1))
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
        ax = fig.add_subplot(n_rows, n_cols, slot + 1)
        dmatrix = results[slot][2]
        indices = np.argsort(results[slot][1])
        sorted_dmatrix = np.take(np.take(dmatrix, indices, 1), indices, 0) 
        ax.matshow(sorted_dmatrix)
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
        ax = fig.add_subplot(n_rows, n_cols, slot + 1)
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
