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

save_figures = True
n_clusters_range = range(2,6)
affinities = 'coarse_fwm'
affinities = 'linear_density'
affinities = 'weighted_distance_rmsd'
sim_path = '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_30structures_sn_122replicas/'
#sim_path = '/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_40structures_sn_109replicas/'
ranks = [1]
ranks = range(len(n_clusters_range))

settings = parse_config_file(sim_path + 'config.cfg')
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

for rank in ranks:
    clusters = []
    for i in range(n_clusters_range[-1]):
        path = sim_path + 'analysis/clustering/rank{}_clustering_{}/cluster{}.pdb'
        path = path.format(rank, affinities, i)
        try:
            clusters.append(load_ensemble_from_pdb(path))
        except IOError:
            break

    fig = plt.figure()

    ax = fig.add_subplot(221)
    for k, c in enumerate(clusters):
        ax.plot(np.array([linear_density(m, bead_radii) for m in c]).mean(0), 
                label='cluster {}'.format(k))
    ax.set_xlabel('(coarse-grained) bead index')
    ax.set_ylabel('local density')
    ax.legend()

    ax = fig.add_subplot(222)
    for k, c in enumerate(clusters):
        ax.plot(np.sort(map(lambda x: -L.log_prob(structures=x.ravel()), c)),
                np.linspace(0,1,len(c),endpoint=False),
                label='cluster {}'.format(k))
    ax.set_xlabel('-log L')
    ax.set_ylabel('ECDF')
    ax.legend()

    ax = fig.add_subplot(223)
    for k, c in enumerate(clusters):
        ax.plot(np.sort(map(radius_of_gyration, c)), np.linspace(0,1,len(c),endpoint=False),
                label='cluster {}'.format(k))
    ax.set_xlabel('radius of gyration')
    ax.set_ylabel('ECDF')
    ax.legend()

    ax = fig.add_subplot(224)
    from csb.bio.utils import average_structure, fit_transform
    for k, c in enumerate(clusters):
        avg_X = average_structure(c)
        aligned_ens = np.array([fit_transform(avg_X, m) for m in c])
        ax.plot(np.sort([rmsd(avg_X, x) for x in aligned_ens]),
                np.linspace(0, 1, len(c), endpoint=False),
                label='cluster {} ({} structures)'.format(k, len(c)))
    ax.legend()
    ax.set_xlabel('RMSD to average structure')
    ax.set_ylabel('ECDF')

    fig.tight_layout()
    if save_figures:
        path = sim_path + 'analysis/clustering/rank{}_clustering_{}/'.format(rank, affinities)
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + 'measures.pdf'.format(rank, affinities))
    else:
        plt.show()
