import numpy as np
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from csb.bio.utils import radius_of_gyration, rmsd

from ensemble_hic.analysis_functions import load_ensemble_from_pdb
from ensemble_hic.setup_functions import make_posterior, parse_config_file
from ensemble_hic.forward_models import EnsembleContactsFWM
from ensemble_hic.likelihoods import Likelihood
from ensemble_hic.error_models import PoissonEM

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/clustering/'))
from clustering_funcs import linear_density

full_norm = True
save_figures = True
n_clusters_range = range(2,6)
affinities = ['coarse_fwm',
              'linear_density',
              'weighted_distance_rmsd'
              ]
sim_path = '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_30structures_sn_122replicas/'
sim_path = '/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_40structures_sn_109replicas/'
ranks = [0]
#ranks = range(len(n_clusters_range))

settings = parse_config_file(sim_path + 'config.cfg')
n_beads = int(settings['general']['n_beads'])
p = make_posterior(settings)
pL = p.likelihoods['ensemble_contacts']
FWM = EnsembleContactsFWM('asdf', 1, pL.forward_model['contact_distances'].value,
                          pL.forward_model.data_points, cutoff=5000.0)
EM = PoissonEM('qwer', FWM.data_points[:,2])
L = Likelihood('asdf', FWM, EM, 1.0)
w = np.ones(1)
alpha = float(settings['forward_model']['alpha'])
L = L.conditional_factory(norm=1.0, weights=w, smooth_steepness=alpha)
fwm_eval = lambda x: L.forward_model(structures=x)
bead_radii = p.priors['nonbonded_prior'].forcefield.bead_radii

def intersect(x, y):
    n_cols = x.shape[1]
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)],
             'formats': n_cols * [x.dtype]}
    dtype = (', '.join([str(x.dtype)] * n_cols))
    C = np.intersect1d(x.view(dtype), y.view(dtype))

    return C.view(x.dtype).reshape(-1, n_cols)


for rank in ranks:
    all_clusters = []
    for affinity in affinities:
        clusters = []
        for i in range(n_clusters_range[-1]):
            path = sim_path + 'analysis/clustering/rank{}_clustering_{}/cluster{}.pdb'
            path = path.format(rank, affinity, i)
            try:
                clusters.append(load_ensemble_from_pdb(path).reshape(-1, n_beads * 3))
            except IOError:
                break
        all_clusters.append(clusters)
    all_clusters_flat = [x for y in all_clusters for x in y]

    fracts = np.array([[len(intersect(x, y)) / float(len(x) + full_norm * len(y)) for x in all_clusters_flat] for y in all_clusters_flat])
    fracts[np.diag_indices(len(fracts))] = 0.0
    labels = [x for y in [['{}, #{}'.format(affinities[i], len(all_clusters[i][j])) for j in range(len(all_clusters[i]))] for i in range(len(affinities))] for x in y]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ms = ax.matshow(fracts)
    cb = fig.colorbar(ms)
    cb.set_label('fraction of common structures')
    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_yticks(np.arange(0, len(labels), 1))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticklabels(labels)
    fig.tight_layout()
        
    if save_figures:
        path = sim_path + 'analysis/clustering/'
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + 'rank{}_affinities_comparison_{}.pdf'.format(rank,
                                                                        'fullnorm' if full_norm else 'asymmetric'))
    else:
        plt.show()
