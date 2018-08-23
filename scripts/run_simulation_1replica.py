import os
import sys
import numpy as np
import ConfigParser

from ensemble_hic.setup_functions import make_replica_schedule, parse_config_file

if True:
    np.random.seed(42)
    rank = 2
    size = 3
    ppath = os.path.expanduser('~/projects/ensemble_hic/')
    config_file = ppath + 'scripts/rao2014/testcfg.cfg'
    settings = parse_config_file(config_file)
    settings['general']['data_file'] = ppath + 'data/rao2014/chr1.txt'
    settings['nonbonded_prior']['bead_radii'] = ppath + 'data/rao2014/chr1_radii.txt'
    settings['general']['n_structures'] = '3'
if True:
    np.random.seed(42)
    rank = 2
    size = 3
    ppath = os.path.expanduser('~/projects/ensemble_hic/')
    spath = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_noii3_20structures_330replicas/'
    # spath = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_it3_20structures_309replicas/'
    config_file = spath + 'config.cfg'
    config_file = sys.argv[1]
    settings = parse_config_file(config_file)
    # settings['general']['data_file'] = ppath + 'data/rao2014/chr1.txt'
    # settings['nonbonded_prior']['bead_radii'] = ppath + 'data/rao2014/chr1_radii.txt'
    # settings['general']['n_structures'] = '3'

# n_replicas = size - 1

# re_params = settings['replica']
# schedule = make_replica_schedule(re_params, n_replicas)
    
from isd2.samplers.gibbs import GibbsSampler

from ensemble_hic.setup_functions import make_posterior, make_subsamplers
from ensemble_hic.setup_functions import setup_initial_state, setup_weights

def make_posterior(settings):

    from isd2.pdf.posteriors import Posterior
    from ensemble_hic.setup_functions import *

    settings = update_ensemble_setting(settings)
    n_beads = int(settings['general']['n_beads'])
    n_structures = int(settings['general']['n_structures'])
    priors = make_priors(settings['nonbonded_prior'],
                         settings['backbone_prior'],
                         settings['sphere_prior'],
                         n_beads, n_structures)
    bead_radii = priors['nonbonded_prior'].forcefield.bead_radii
    likelihood = make_likelihood2(settings['forward_model'],
                                  settings['general']['error_model'],
                                  settings['data_filtering'],
                                  settings['general']['data_file'],
                                  n_structures, bead_radii)
    if 'norm' in settings['general']['variables'].split(','):
        priors.update(norm_prior=make_norm_prior(settings['norm_prior'],
                                                 likelihood, n_structures))
    full_posterior = Posterior({likelihood.name: likelihood}, priors)

    return make_conditional_posterior(full_posterior, settings)

def make_likelihood2(forward_model_params, error_model, data_filtering_params,
                     data_file, n_structures, bead_radii):

    from ensemble_hic.forward_models import EnsembleContactsFWM
    from ensemble_hic.likelihoods import Likelihood

    disregard_lowest = data_filtering_params['disregard_lowest']
    ignore_sequential_neighbors = int(data_filtering_params['ignore_sequential_neighbors'])
    include_zero_counts = data_filtering_params['include_zero_counts']
    data = np.loadtxt(data_file, dtype=int)
    if include_zero_counts == 'False':
        data = data[data[:,2] > 0]
    data = data[np.argsort(data[:,2])]
    data = data[int(disregard_lowest * len(data)):]
    data = data[np.abs(data[:,0] - data[:,1]) > ignore_sequential_neighbors]
    #data = data[:-2]
    cd_factor = float(forward_model_params['contact_distance_factor'])
    contact_distances = (bead_radii[data[:,0]] + bead_radii[data[:,1]]) * cd_factor
        
    FWM = EnsembleContactsFWM('fwm', n_structures, contact_distances,
                              data_points=data)

    if error_model == 'poisson':
        from ensemble_hic.error_models import PoissonEM
        EM = PoissonEM('ensemble_contacts_em', data[:,2])
    else:
        raise(NotImplementedError)

    L = Likelihood('ensemble_contacts', FWM, EM, 1.0)
    L = L.conditional_factory(smooth_steepness=forward_model_params['alpha'])
    
    return L



settings['initial_state']['weights'] = setup_weights(settings)
# settings['structures_hmc']['timestep'] = np.loadtxt(spath + 'statistics/mcmc_stats.txt')[-1,-1]
posterior = make_posterior(settings)
# for replica_parameter in schedule:
#     posterior[replica_parameter].set(schedule[replica_parameter][rank - 1])

initial_state = setup_initial_state(settings['initial_state'], posterior)
# initial_state = np.load(spath + 'samples/samples_replica330_33000-34000.pickle')[-1]
if not 'norm' in initial_state.variables:
    posterior['norm'].set(np.max(posterior.likelihoods['ensemble_contacts'].error_model.data) / float(settings['general']['n_structures']))
# initial_state._variables.pop('norm')
# posterior = posterior.conditional_factory(norm=750.0)
subsamplers = make_subsamplers(posterior, initial_state.variables,
                               settings['structures_hmc'],
                               settings['weights_hmc'])

sampler = GibbsSampler(pdf=posterior, state=initial_state,
                       subsamplers=subsamplers)    

if False:
    from copy import deepcopy
    n = 0
    n_acc = 0
    n_samples = 200
    samples = []
    for i in range(n_samples):
        samples.append(deepcopy(sampler.sample()))
        n +=1
        n_acc += sampler.subsamplers['structures'].last_draw_stats['structures'].accepted
        if i > 0 and i % 5 == 0:
            print "{} / {} -- p_acc: {:.2f}".format(n-1, n_samples, n_acc / float(n))
        
if False:
    n_structures = int(settings['general']['n_structures'])
    md = posterior.likelihoods['ensemble_contacts'].forward_model(**samples[-1].variables)
    d = posterior.likelihoods['ensemble_contacts'].forward_model.data_points

    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3,2)
    for i, ax in enumerate(axes.ravel()):
        exec('ax{} = ax'.format(i+1))
    ax1.scatter(d[:,2], md)
    ax1.set_xlabel('experimental counts')
    ax1.set_xlabel('mock counts')
    ax1.legend()
    ax1.set_aspect('equal')

    if 'norm' in samples[-1].variables:
        ax2.hist([x.variables['norm'] for x in samples[10:]])
        ax2.set_xlabel('$\gamma$')

    E_nb = np.array(map(lambda x: -posterior.priors['nonbonded_prior'].log_prob(structures=x.variables['structures']), samples[10:]))
    ax3.hist(E_nb / n_structures)
    ax3.set_xlabel('$E_{nb}$')

    E_bb = np.array(map(lambda x: -posterior.priors['backbone_prior'].log_prob(structures=x.variables['structures']), samples[10:]))
    ax4.hist(E_bb / n_structures)
    ax4.set_xlabel('$E_{bb}$')

    from csb.bio.utils import radius_of_gyration
    X = np.array([x.variables['structures'].reshape(n_structures, -1, 3)
                  for x in samples])
    rgs = map(radius_of_gyration, X.reshape(-1, 308,3))
    ax5.hist(rgs)
    ax5.set_xlabel('$r_{gyr}$')

    fig.tight_layout()
    
    plt.show()

if False:
    from csb.bio.utils import distance_matrix

    dms = np.array(map(distance_matrix, X.reshape(-1, 308, 3)))
    mask = dms > 1
    nbm = 0.5 * 500 * (dms - 1) ** 4
    nbm[mask] = 0
    for i in range(len(nbm)):
        nbm[i][np.diag_indices(308)] = 0

    fwm = posterior.likelihoods['ensemble_contacts'].forward_model
    mds = np.array([fwm(**x.variables) for x in samples])

    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.scatter(mds.sum(0),
                np.sum([[nbm[i,j,k] for i in range(len(nbm))]
                        for j,k in fwm.data_points[:,:2]], 1)
                )

    plt.show()
