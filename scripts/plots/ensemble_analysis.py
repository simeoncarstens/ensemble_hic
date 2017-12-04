import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering
from csb.bio.utils import rmsd, distance_matrix, radius_of_gyration, average_structure, scale_and_fit
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg, write_ensemble, write_VMD_script, write_pymol_script

step = 10
if not True:
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
else:
    from protlib import parseMultiModelFile
    ens = np.concatenate((parseMultiModelFile('/home/simeon/test/t0/cluster0.pdb', 70, 1037),
                          parseMultiModelFile('/home/simeon/test/t0/cluster1.pdb', 70, 363)))
    ens = ens[np.random.choice(np.arange(len(ens)), 300, replace=False)]
    ens = ens[None,:]
    config_file = '/home/simeon/test/clustering/config.cfg'
    settings = parse_config_file(config_file)
    settings['nonbonded_prior']['bead_radii'] = '/home/simeon/projects/ensemble_hic/scripts/bau2011/bead_radii.txt'
    settings['general']['data_file'] = '/home/simeon/projects/ensemble_hic/data/bau2011/GM12878_processed.txt'
    output_folder = '/home/simeon/test/clustering/'
p = make_posterior(settings)
bead_radii = p.priors['nonbonded_prior'].bead_radii
L = p.likelihoods['ensemble_contacts']
    
save_figures = not False
save_ensembles = True
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

def weighted_distance_rmsd_affinities(ens):
    brsum = np.add.outer(bead_radii, bead_radii) ** 2
    brsum[np.diag_indices(len(brsum))] = 0.0
    brsum = squareform(brsum)
    brsum /= sum(brsum)
    
    rmsds = squareform([np.sum(brsum * (pdist(ens[i]) - pdist(ens[j])) ** 2)
                        for i in range(len(ens))
                        for j in range(i+1, len(ens))])

    return np.exp(-rmsds / 100000.0)

def fwm_affinities(ens):
    from ensemble_hic.forward_models import EnsembleContactsFWM
    FWM = EnsembleContactsFWM('asdf', 1, L.forward_model['contact_distances'].value,
                              L.forward_model.data_points, cutoff=5000.0)
    sel = np.where(np.abs(FWM.data_points[:,0] - FWM.data_points[:,1]) > 15)[0]
    print len(sel)
    w=np.ones(1)

    fwm_eval = lambda x: FWM(structures=x.ravel(), weights=w, norm=1, 
                             smooth_steepness=float(settings['forward_model']['alpha']))
    dp2 = FWM.data_points[sel,2] ** 2
    rmsds = squareform([np.sum(dp2 * (fwm_eval(ens[i])[sel] - fwm_eval(ens[j])[sel]) ** 2)
                        for i in range(len(ens))
                        for j in range(i+1, len(ens))])

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
    n_clusters_range = range(2, 5)

    affinities = weighted_distance_rmsd_affinities
    aff_str = 'wdistrmsd'

    affinities = fwm_affinities
    aff_str = 'fwm'    
    
    ens_flatter = ens.reshape(ens.shape[0] * ens.shape[1], -1, 3)
    res = perform_clustering(ens_flatter,
                             n_clusters_range, affinities)
    sh_scores, all_labels, dmatrix = res
    n = min(n_clusters_range)
    while n in n_clusters_range and save_ensembles:
        rank = n - min(n_clusters_range)
        labels = get_labels(rank, all_labels, sh_scores)
        outf = figures_folder + 'rank{}_clustering_{}/'.format(rank, aff_str)
        if not os.path.exists(outf):
            os.makedirs(outf)
        fig = plt.figure()
        rog_ax = fig.add_subplot(121)
        rmsd_ax = fig.add_subplot(122)
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
            rog_ax.plot(np.sort(map(radius_of_gyration, aligned_ens)), 
                        np.linspace(0 ,1 ,len(members), endpoint=False),
                        label='cluster #{} ({} models)'.format(k, len(members)))
            rmsd_ax.plot(np.sort(map(lambda x: rmsd(avg_X, x), aligned_ens)),
                         np.linspace(0 ,1 ,len(members), endpoint=False), 
                         label='cluster #{} ({} models)'.format(k, len(members)))
            
            from ensemble_hic.forward_models import EnsembleContactsFWM
            FWM = EnsembleContactsFWM('asdf', 1, L.forward_model['contact_distances'].value,
                                      L.forward_model.data_points, cutoff=5000.0)
            sel = np.where(np.abs(FWM.data_points[:,0] - FWM.data_points[:,1]) > 15)[0]
            w=np.ones(1)
            fwm_eval = lambda x: FWM(structures=x.ravel(), weights=w, norm=1, 
                                     smooth_steepness=float(settings['forward_model']['alpha']))
            fwmfig = plt.figure()
            for i in range(0, min(len(members), 20)):
                fwmax = fwmfig.add_subplot(4,5,i+1)
                m = np.zeros((70,70))
                m[FWM.data_points[:,0], FWM.data_points[:,1]] = fwm_eval(members[i])
                from ensemble_hic import kth_diag_indices
                m[kth_diag_indices(m, 0)] = 0.0
                m[kth_diag_indices(m, 1)] = 0.0
                m[kth_diag_indices(m, -1)] = 0.0
                m[kth_diag_indices(m, 2)] = 0.0
                m[kth_diag_indices(m, -2)] = 0.0
                fwmax.matshow(m + m.T)
                fwmax.set_xticks([])
                fwmax.set_yticks([])
            fwmfig.savefig(outf + 'c{}md.pdf'.format(k))

     
        rog_ax.legend()
        rog_ax.set_xlabel('radius of gyration')
        rog_ax.set_ylabel('ECDF')
        rmsd_ax.legend()
        rmsd_ax.set_xlabel('RMSD to average structure')
        rog_ax.set_ylabel('ECDF')
        fig.tight_layout()
        plt.savefig(outf + 'measures.pdf')
        
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

    fig.tight_layout()
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
