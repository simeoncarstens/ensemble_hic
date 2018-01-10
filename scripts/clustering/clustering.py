import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering
from csb.bio.utils import rmsd, average_structure, scale_and_fit
from scipy.spatial.distance import pdist, squareform
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg, write_ensemble, write_VMD_script, write_pymol_script

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/clustering/'))
from clustering_funcs import *

if True:
    config_file = sys.argv[1]
    n_processes = int(sys.argv[2])
    n_clusters_range = range(int(sys.argv[3]), int(sys.argv[4]))
    affinities = sys.argv[5]
    step = int(sys.argv[6])
    settings = parse_config_file(config_file)
    output_folder = settings['general']['output_folder']
    samples = load_samples_from_cfg(config_file)[::step]
    
    if 'weights' in samples[-1].variables:
        weights = np.array([x.variables['weights'] for x in samples.ravel()])
    if 'norm' in samples[-1].variables:
        norms = np.array([x.variables['norm'] for x in samples.ravel()])
    
    ens = np.array([sample.variables['structures'].reshape(-1, 70, 3)
                    for sample in samples])
p = make_posterior(settings)
bead_radii = p.priors['nonbonded_prior'].forcefield.bead_radii
L = p.likelihoods['ensemble_contacts']
FWM = L.forward_model
alpha = float(settings['forward_model']['alpha'])

figures_folder = output_folder + 'analysis/clustering/'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

ens_flatter = ens.reshape(ens.shape[0] * ens.shape[1], -1, 3)

if True:
    ## cluster all structures

    if affinities == 'coarse_fwm':
        affinities = lambda ens: coarse_fwm_affinities(ens,
                                                       FWM['contact_distances'].value,
                                                       FWM.data_points,
                                                       alpha)
        aff_str = 'coarse_fwm'
    if affinities == 'linear_density':    
        affinities = lambda ens: linear_density_affinities(ens, bead_radii)
        aff_str = 'linear_density'    
    if affinities == 'weighted_distance_rmsd':    
        affinities = lambda ens: weighted_distance_rmsd_affinities(ens, bead_radii)
        aff_str = 'weighted_distance_rmsd'    

    affinities_results = affinities(ens_flatter)
    res = perform_clustering(n_clusters_range, affinities_results)
    sh_scores, all_labels, dmatrix = res
    n = min(n_clusters_range)
    while n in n_clusters_range:
        rank = n - min(n_clusters_range)
        labels = get_labels(rank, all_labels, sh_scores)
        outf = figures_folder + 'rank{}_clustering_{}/'.format(rank, aff_str)
        if not os.path.exists(outf):
            os.makedirs(outf)
        for k in xrange(n):
            members = ens_flatter[labels == k]
            if len(members) == 0:
                continue
            # avg_X = average_structure(members)
            # Rt = [scale_and_fit(avg_X, Y)[:2] for Y in members]
            # aligned_ens = np.array([np.dot(members[i], R.T) + t for i, (R, t)
            #                         in enumerate(Rt)])
            aligned_ens = members
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

    ax = fig.add_subplot(222)
    labels = get_labels(0, all_labels, sh_scores)
    indices = np.argsort(labels)
    sorted_dmatrix = np.take(np.take(dmatrix, indices, 1), indices, 0)
    ax.matshow(sorted_dmatrix, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('{} clusters'.format(len(set(labels))))

    ax = fig.add_subplot(224)
    labels = get_labels(1, all_labels, sh_scores)
    indices = np.argsort(labels)
    sorted_dmatrix = np.take(np.take(dmatrix, indices, 1), indices, 0)
    ax.matshow(sorted_dmatrix, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('{} clusters'.format(len(set(labels))))

    plt.savefig(figures_folder + aff_str + '_all_structures.pdf')
    
