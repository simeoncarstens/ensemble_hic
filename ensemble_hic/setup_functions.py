import numpy as np

from rexfw.statistics.writers import ConsoleStatisticsWriter

def parse_config_file(config_file):

    import ConfigParser

    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    def config_section_map(section):
        dict1 = {}
        options = config.options(section)
        for option in options:
            try:
                dict1[option] = config.get(section, option)
            except:
                dict1[option] = None
        return dict1

    return {section: config_section_map(section) for section in config.sections()}    

def make_posterior(settings):

    from isd2.pdf.posteriors import Posterior
    from ensemble_hic.setup_functions import make_priors, make_likelihood

    n_beads = int(settings['general']['n_beads'])
    n_structures = int(settings['general']['n_structures'])
    priors = make_priors(settings['nonbonded_prior'],
                         settings['backbone_prior'],
                         settings['sphere_prior'],
                         n_beads, n_structures)
    bead_radii = priors['nonbonded_prior'].bead_radii
    likelihood = make_likelihood(settings['forward_model'],
                                 settings['general']['error_model'],
                                 settings['data_filtering'],
                                 settings['general']['data_file'],
                                 n_structures, bead_radii)

    full_posterior = Posterior({likelihood.name: likelihood}, priors)

    return make_conditional_posterior(full_posterior, settings)

def setup_weights(settings):

    weights_string = settings['initial_state']['weights']
    n_structures = int(settings['general']['n_structures'])
    try:
        weights = float(weights_string)
        weights = np.ones(n_structures) * weights
    except:
        weights = np.loadtxt(weights_string, dtype=float)

    return weights

def make_replica_schedule(replica_params, n_replicas):

    lambda_min = float(replica_params['lambda_min'])
    lambda_max = float(replica_params['lambda_max'])
    beta_min = float(replica_params['beta_min'])
    beta_max = float(replica_params['beta_max'])
    separate_prior_annealing = True if replica_params['separate_prior_annealing'] == 'True' else False

    if separate_prior_annealing:
        beta_chain = np.arange(0, np.floor(n_replicas / 2))
        lambda_chain = np.arange(np.floor(n_replicas / 2), n_replicas)
        lambdas = np.concatenate((np.zeros(len(beta_chain)) + lambda_min,
                                  np.linspace(lambda_min, lambda_max,
                                              len(lambda_chain))))
        betas = np.concatenate((np.linspace(beta_min, beta_max,
                                            len(beta_chain)),
                                np.zeros(len(lambda_chain)) + beta_max))
        schedule = {'lammda': lambdas, 'beta': betas}
    else:
        schedule = {'lammda': np.linspace(lambda_min, lambda_max, n_replicas),
                    'beta': np.linspace(beta_min, beta_max, n_replicas)}

    return schedule


def make_subsamplers(posterior, initial_state,
                     structures_hmc_params, weights_hmc_params):

    from .hmc import HMCSampler

    p = posterior
    variables = initial_state.keys()
    structures_tl = int(structures_hmc_params['trajectory_length'])
    structures_timestep = float(structures_hmc_params['timestep'])
    structures_adaption = structures_hmc_params['timestep_adaption']
    structures_adaption = True if structures_adaption == 'True' else False
    weights_tl = int(weights_hmc_params['trajectory_length'])
    weights_timestep = float(weights_hmc_params['timestep'])
    weights_adaption = weights_hmc_params['timestep_adaption']
    weights_adaption = True if weights_adaption == 'True' else False

    if not structures_adaption or not weights_adaption:
        raise NotImplementedError('At the moment, timestep cannot be switched off!')

    xpdf = p.conditional_factory(**{var: value for (var, value)
                                    in initial_state.iteritems()
                                    if not var == 'structures'})
    structures_sampler = HMCSampler(xpdf,
                                    initial_state['structures'],
                                    structures_timestep, structures_tl)

    subsamplers = dict(structures=structures_sampler)

    if 'norm' in variables:
        from .error_models import PoissonEM
        if isinstance(p.likelihoods['ensemble_contacts'].error_model,
                      PoissonEM):
            from .npsamplers import NormGammaSampler
            norm_sampler = NormGammaSampler()
            subsamplers.update(norm=norm_sampler)
        else:
            raise NotImplementedError('Norm sampling only implemented' +
                                      'for Poisson error model!')
    if 'weights' in variables:
        raise NotImplementedError('Weights sampling not implemented yet')

    return subsamplers
    

def make_elongated_structures(bead_radii, n_structures):

    X = [bead_radii[0]]
    for i in range(len(bead_radii) -1):
        X.append(X[-1] + bead_radii[i+1] + bead_radii[i])
    X = np.array(X) - np.mean(X)
    
    X = np.array([X, np.zeros(len(bead_radii)),
                  np.zeros(len(bead_radii))]).T[None,:]
    X = X.repeat(n_structures, 0).ravel().astype(float)

    return X    

def setup_initial_state(initial_state_params, posterior):

    from isd2.samplers import ISDState

    p = posterior
    n_structures = p.likelihoods['ensemble_contacts'].forward_model.n_structures
    structures = initial_state_params['structures']
    norm = initial_state_params['norm']
    variables = p.variables

    if structures == 'elongated':
        bead_radii = posterior.priors['nonbonded_prior'].bead_radii
        structures = make_elongated_structures(bead_radii, n_structures)
        structures += np.random.normal(scale=0.5, size=structures.shape)
    else:
        try:
            structures = np.load(structures)
        except:
            raise ValueError('Couldn\'t load initial structures '
                             'from file {}!'.format(structures))

    init_state = ISDState({'structures': structures})

    if 'weights' in variables:
        init_state.update_variables(weights=weights)
    if 'norm' in variables:
        init_state.update_variables(norm=norm)

    return init_state
    

def make_conditional_posterior(posterior, settings):

    variables = settings['general']['variables'].split(',')
    variables = [x.strip() for x in variables]
    p = posterior

    if 'norm' in variables and 'weights' in variables:
        raise NotImplementedError('Can\'t estimate both norm and weights!')
    elif 'norm' in variables:
        n_structures = p.likelihoods['ensemble_contacts'].forward_model.n_structures
        return p.conditional_factory(weights=np.ones(n_structures))
    elif 'weights' in variables:
        return p.conditional_factory(norm=1.0)
    else:
        return p.conditional_factory(norm=settings['initial_state']['norm'],
                                     weights=settings['initial_state']['weights'])
    
def make_priors(nonbonded_prior_params, backbone_prior_params,
                sphere_prior_params, n_beads, n_structures):

    from .nonbonded_prior import NonbondedPrior
    from .backbone_prior import BackbonePrior
    from .sphere_prior import SpherePrior

    try:
        bead_radius = float(nonbonded_prior_params['bead_radii'])
        bead_radii = np.ones(n_beads) * bead_radius
    except:
        bead_radii = np.loadtxt(nonbonded_prior_params['bead_radii'],
                                dtype=float)
        
    bb_ll, bb_ul = np.zeros(n_beads - 1), bead_radii[:-1] + bead_radii[1:]
    
    NBP = NonbondedPrior('nonbonded_prior', bead_radii=bead_radii,
                         force_constant=nonbonded_prior_params['force_constant'],
                         beta=1.0, n_structures=n_structures)
    BBP = BackbonePrior('backbone_prior',
                        lower_limits=bb_ll, upper_limits=bb_ul,
                        k_bb=float(backbone_prior_params['force_constant']),
                        n_structures=n_structures)
    SP = SpherePrior('sphere_prior',
                     sphere_radius=float(sphere_prior_params['radius']),
                     sphere_k=float(sphere_prior_params['force_constant']),
                     n_structures=n_structures)
    priors = {NBP.name: NBP, BBP.name: BBP, SP.name: SP}

    return priors

def make_likelihood(forward_model_params, error_model, data_filtering_params,
                    data_file, n_structures, bead_radii):

    from .forward_models import EnsembleContactsFWM
    from .likelihoods import Likelihood

    disregard_lowest = data_filtering_params['disregard_lowest']
    ignore_sequential_neighbors = int(data_filtering_params['ignore_sequential_neighbors'])
    data = np.loadtxt(data_file, dtype=int)
    data = data[np.argsort(data[:,2])]
    data = data[int(disregard_lowest * len(data)):]
    data = data[np.abs(data[:,0] - data[:,1]) > ignore_sequential_neighbors]
    cd_factor = float(forward_model_params['contact_distance_factor'])
    contact_distances = (bead_radii[data[:,0]] + bead_radii[data[:,1]]) * cd_factor
        
    FWM = EnsembleContactsFWM('fwm', n_structures, contact_distances,
                              cutoff=10000.0, data_points=data)

    if error_model == 'poisson':
        from .error_models import PoissonEM
        EM = PoissonEM('ensemble_contacts_em', data[:,2])
    else:
        raise(NotImplementedError)

    L = Likelihood('ensemble_contacts', FWM, EM, 1.0)
    L = L.conditional_factory(smooth_steepness=forward_model_params['alpha'])
    
    return L

def setup_default_re_master(n_replicas, sim_path, comm):

    from rexfw.remasters import ExchangeMaster
    from rexfw.statistics import Statistics, REStatistics
    from rexfw.statistics.writers import StandardConsoleREStatisticsWriter, StandardFileMCMCStatisticsWriter, StandardFileREStatisticsWriter, StandardFileREWorksStatisticsWriter, StandardConsoleMCMCStatisticsWriter, StandardConsoleMCMCStatisticsWriter
    from rexfw.convenience import create_standard_RE_params
    from rexfw.convenience.statistics import create_standard_averages, create_standard_works, create_standard_stepsizes, create_standard_heats

    replica_names = ['replica{}'.format(i) for i in range(1, n_replicas + 1)]
    params = create_standard_RE_params(n_replicas)
        
    from rexfw.statistics.averages import REAcceptanceRateAverage, MCMCAcceptanceRateAverage
    from rexfw.statistics.logged_quantities import SamplerStepsize
    
    local_pacc_avgs = [MCMCAcceptanceRateAverage(r, 'structures')
                       for r in replica_names]
    # local_pacc_avgs += [MCMCAcceptanceRateAverage(r, 'weights')
    #                     for r in replica_names]
    re_pacc_avgs = [REAcceptanceRateAverage(replica_names[i], replica_names[i+1]) 
                    for i in range(len(replica_names) - 1)]
    stepsizes = [SamplerStepsize(r, 'structures') for r in replica_names]
    # stepsizes += [SamplerStepsize(r, 'weights') for r in replica_names]
    # stepsizes += [SamplerStepsize(r, 'k2') for r in replica_names]
    works = create_standard_works(replica_names)
    heats = create_standard_heats(replica_names)
    stats_path = sim_path + 'statistics/'
    stats_writers = [StandardConsoleMCMCStatisticsWriter(['structures',
                                                          #'weights'
                                                          #'k2',
                                                          ],
                                                         ['acceptance rate',
                                                          'stepsize']),
                     StandardFileMCMCStatisticsWriter(stats_path + '/mcmc_stats.txt',
                                                      ['structures',
                                                       #'weights'
                                                       ],
                                                      ['acceptance rate', 'stepsize'])
                    ]
    stats = Statistics(elements=local_pacc_avgs + stepsizes, 
                       stats_writer=stats_writers)
    re_stats_writers = [StandardConsoleREStatisticsWriter(),
                        StandardFileREStatisticsWriter(stats_path + 're_stats.txt',
                                                       ['acceptance rate'])]
    works_path = sim_path + 'works/'
    works_writers = [StandardFileREWorksStatisticsWriter(works_path)]
    re_stats = REStatistics(elements=re_pacc_avgs,
                            work_elements=works, heat_elements=heats,
                            stats_writer=re_stats_writers,
                            works_writer=works_writers)
    
    master = ExchangeMaster('master0', replica_names, params, comm=comm, 
                            sampling_statistics=stats, swap_statistics=re_stats)

    return master
