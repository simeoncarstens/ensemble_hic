"""
A lot of helper functions to set up the posterior distribution,
samplers and replica exchange scheme
"""
import numpy as np

from rexfw.statistics.writers import ConsoleStatisticsWriter

def parse_config_file(config_file):
    """
    Parses a config file consisting of several sections.
    I think I adapted this from the ConfigParser docs.

    :param config_file: config file name
    :type config_file: str

    :returns: a nested dictionary with sections and section content
    :rtype: dict of dicts
    """
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

def update_ensemble_setting(settings):
    """
    Copies ensemble setting (either 'boltzmann' or 'tsallis') from
    'replica' replica section in config dicts to the nonbonded section

    If no ensemble is set in the 'replica' section, 'boltzmann' is
    assumed.

    :param settings: settings specified in 'replica' section of
                     a config file
    :type settings: dict

    :returns: updated settings
    :rtype: dict of dicts
    """
    rps = settings['replica']
    if not 'ensemble' in rps or rps['ensemble'] == 'boltzmann':
        settings['nonbonded_prior'].update(ensemble='boltzmann')
    elif rps['ensemble'] == 'tsallis':
        settings['nonbonded_prior'].update(ensemble='tsallis')
    else:
        msg = 'ensemble has to be either not present in cfg file,'+\
               ' or set to \'tsallis\' or \'boltzmann\''
        raise ValueError(msg)

    return settings    

def make_posterior(settings):
    """
    Builds a posterior object from settings

    You usually want to do something like
        p = make_posterior(parse_config_file('config.cfg'))

    :param settings: settings specified in 'replica' section of
                     a config file
    :type settings: dict

    :returns: a posterior object
    :rtype: :class:`binf.pdf.posteriors.Posterior`
    """
    from binf.pdf.posteriors import Posterior

    settings = update_ensemble_setting(settings)
    n_beads = int(settings['general']['n_beads'])
    n_structures = int(settings['general']['n_structures'])
    priors = make_priors(settings['nonbonded_prior'],
                         settings['backbone_prior'],
                         settings['sphere_prior'],
                         n_beads, n_structures)
    bead_radii = priors['nonbonded_prior'].forcefield.bead_radii
    likelihood = make_likelihood(settings['forward_model'],
                                 settings['general']['error_model'],
                                 settings['data_filtering'],
                                 settings['general']['data_file'],
                                 n_structures, bead_radii)
    if 'norm' in settings['general']['variables'].split(','):
        priors.update(norm_prior=make_norm_prior(settings['norm_prior'],
                                                 likelihood, n_structures))
    full_posterior = Posterior({likelihood.name: likelihood}, priors)

    return make_conditional_posterior(full_posterior, settings)

def make_norm_prior(norm_prior_settings, likelihood, n_structures):
    """
    Makes the Gamma prior object for the scaling parameter

    Shape and rate of the Gamma distribution are set to rather broad
    values depending on the average number of counts in the data

    :param norm_prior_settings: settings for the scaling factor prior
                                as specified in a config file
    :type norm_prior_settings: dict

    :param likelihood: a likelihood object from which the count data
                       can be retrieved
    :type likelihood: :class:`.Likelihood`

    :param n_structures: number of ensemble members
    :type n_structures: int

    :returns: a scaling factor prior object
    :rtype: :class:`.NormGammaPrior`
    """
    from .gamma_prior import NormGammaPrior

    shape = norm_prior_settings['shape']
    rate = norm_prior_settings['rate']
    if shape == rate == 'auto':
        rate = 1.0 / n_structures
        dp = likelihood.forward_model.data_points[:,2]
        shape = np.mean(dp[dp > 0]) / float(n_structures)
    else:
        shape = float(shape)
        rate = float(rate)

    return NormGammaPrior(shape, rate)

def make_marginalized_posterior(settings):

    from binf.pdf.posteriors import Posterior

    settings = update_ensemble_setting(settings)
    n_beads = int(settings['general']['n_beads'])
    n_structures = int(settings['general']['n_structures'])
    priors = make_priors(settings['nonbonded_prior'],
                         settings['backbone_prior'],
                         settings['sphere_prior'],
                         n_beads, n_structures)
    from .gamma_prior import NormGammaPrior
    if 'norm_prior' in settings:
        shape = float(settings['norm_prior']['shape'])
        rate = float(settings['norm_prior']['rate'])
    else:
        shape = 0.1
        rate = 0.1
    priors.update(norm_prior=NormGammaPrior(shape,rate))
    bead_radii = priors['nonbonded_prior'].bead_radii
    likelihood = make_likelihood(settings['forward_model'],
                                 settings['general']['error_model'],
                                 settings['data_filtering'],
                                 settings['general']['data_file'],
                                 n_structures, bead_radii)

    from binf.pdf import AbstractISDPDF
    from .marginalized_posterior_c import calculate_gradient
    alpha = priors['norm_prior']['shape'].value
    beta = priors['norm_prior']['rate'].value

    class MyPdf(AbstractISDPDF):

        def __init__(self, likelihood, priors, lammda=1.0, beta=1.0):

            from csb.statistics.pdf.parameterized import Parameter
            super(MyPdf, self).__init__()

            self.L = likelihood
            self.Ps = priors
            self._register('lammda')
            self['lammda'] = Parameter(lammda, 'lammda')
            self._register('beta')
            self['beta'] = Parameter(beta, 'lammda')
            self._register_variable('structures', differentiable=True)

        def _evaluate_log_prob(self, structures):
            d = self.L.error_model.data
            md = self.L.forward_model(structures=structures, norm=1.0,
                                          weights=np.ones(n_structures))
            
            return self['lammda'].value * (np.sum(d * np.log(md)) - (d.sum() + alpha) * np.log(md.sum() + beta)) + self['beta'].value * self.Ps['nonbonded_prior'].log_prob(structures=structures) + self.Ps['backbone_prior'].log_prob(structures=structures)

        def _evaluate_gradient(self, structures):
            L = self.L
            b = self['beta'].value
            return calculate_gradient(structures.reshape(n_structures, n_beads, 3),
                                      L['smooth_steepness'].value,
                                      L.forward_model['contact_distances'].value,
                                      L.forward_model.data_points,
                                      alpha, beta) * self['lammda'].value \
                    + self.Ps['nonbonded_prior'].gradient(structures=structures) * b\
                    + self.Ps['backbone_prior'].gradient(structures=structures)

        def clone(self):
            copy = self.__class__(self.L,
                                  self.Ps,
                                  self['lammda'].value,
                                  self['beta'].value)
            
            copy.fix_variables(**{p: self[p].value for p in self.parameters
                                  if not p in copy.parameters})

            return copy
            
    return MyPdf(likelihood, priors, 1.0, 1.0)
    

def setup_weights(settings):
    """
    Sets up the vector of initial weights

    :param settings: simulation settings as specified in a
                     config file
    :type settings: dict of dicts

    :returns: a weights vector
    :rtype: :class:`numpy.ndarray`
    """
    weights_string = settings['initial_state']['weights']
    n_structures = int(settings['general']['n_structures'])
    try:
        weights = float(weights_string)
        weights = np.ones(n_structures) * weights
    except:
        weights = np.loadtxt(weights_string, dtype=float)

    return weights

def expspace(min, max, a, N):
    """
    Helper function which creates an array of exponentially spaced values

    I use this to create temperature schedules for 
    replica exchange simulations.
    
    :param min: minimum value
    :type min: float

    :param max: maximum value
    :type max: float

    :param a: rate parameter
    :type a: float

    :param N: desired # of values (including min and max)
    :type N: int

    :returns: array of exponentially spaced numbers
    :rtype: :class:`numpy.ndarray`
    """
    g = lambda n: (max - min) / (np.exp(a*(N-1.0)) - 1.0) * (np.exp(a*(n-1.0)) - 1.0) + float(min)
    
    return np.array(map(g, np.arange(1, N+1)))

def make_replica_schedule(replica_params, n_replicas):
    """
    Makes a replica exchange schedule from settings specified in
    a config file.

    You can either have a linear or an exponential schedule
    and a separate prior annealing chain or not. You can also
    load a schedule from a Python pickle. It has to be a
    dict with the keys being the tempered parameters and the
    values the schedule for that parameter.
    But the latter option is currently handled in the run_simulation.py
    script.

    :param replica_params: replica settings as specified in a config file
    :type replica_params: dict

    :param n_replicas: # of replicas
    :type n_replicas: int

    :returns: a replica exchange schedule
    :rtype: dict, e.g., {'beta': np.array([0, 0.33, 0.66, 1.0])}
    """
    l_min = float(replica_params['lambda_min'])
    l_max = float(replica_params['lambda_max'])
    b_min = float(replica_params['beta_min'])
    b_max = float(replica_params['beta_max'])
    
    if replica_params['schedule'] == 'linear':
        if replica_params['separate_prior_annealing'] == 'True':
            separate_prior_annealing = True
        else:
            separate_prior_annealing = False

        if separate_prior_annealing:
            b_chain = np.arange(0, np.floor(n_replicas / 2))
            l_chain = np.arange(np.floor(n_replicas / 2), n_replicas)
            lambdas = np.concatenate((np.zeros(len(b_chain)) + l_min,
                                      np.linspace(l_min, l_max,
                                                  len(l_chain))))
            betas = np.concatenate((np.linspace(b_min, b_max,
                                                len(b_chain)),
                                    np.zeros(len(l_chain)) + b_max))
            schedule = {'lammda': lambdas, 'beta': betas}
        else:
            schedule = {'lammda': np.linspace(l_min, l_max, n_replicas),
                        'beta': np.linspace(b_min, b_max, n_replicas)}
    elif replica_params['schedule'] == 'exponential':
        l_rate = float(replica_params['lambda_rate'])
        b_rate = float(replica_params['beta_rate'])
        schedule = {'lammda': expspace(l_min, l_max, l_rate, n_replicas),
                    'beta':   expspace(b_min, b_max, b_rate, n_replicas)}
    else:
        msg = 'Schedule has to be either a file name, ' + \
              '\'lambda_beta\', or \'exponential\''
        raise ValueError(msg)

    return schedule

def make_subsamplers(posterior, initial_state,
                     structures_hmc_params, weights_hmc_params):
    """
    Makes a dictionary of (possibly MCMC) samplers for all variables

    :param posterior: posterior distribution you want to sample
    :type posterior: :class:`binf.pdf.posteriors.Posterior

    :param initial_state: intial state
    :type initial_state: :class:`binf.samplers.BinfState`

    :param structures_hmc_params: settings for the structures HMC
                                  sampler as specified in a config file
    :type structures_hmc_params: dict

    :param weights_hmc_params: settings for the weights HMC
                               sampler as specified in a config file
    :type weights_hmc_params: dict

    :returns: a dictionary with the keys being the variables and
              the values the corresponding samplers over which
              a Gibbs sampler eventually will iterate
    :rtype: dict
    """
    from binf.samplers.hmc import HMCSampler

    p = posterior
    variables = initial_state.keys()
    structures_tl = int(structures_hmc_params['trajectory_length'])
    structures_timestep = float(structures_hmc_params['timestep'])
    s_adaption_limit = int(structures_hmc_params['adaption_limit'])
    weights_tl = int(weights_hmc_params['trajectory_length'])
    weights_timestep = float(weights_hmc_params['timestep'])
    w_adaption_limit = int(weights_hmc_params['adaption_limit'])

    xpdf = p.conditional_factory(**{var: value for (var, value)
                                    in initial_state.iteritems()
                                    if not var == 'structures'})
    structures_sampler = HMCSampler(xpdf,
                                    initial_state['structures'],
                                    structures_timestep, structures_tl,
                                    variable_name='structures',
                                    timestep_adaption_limit=s_adaption_limit)

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
    """
    Makes a set of fully elongated structures

    :param bead_radii: bead radii for each bead
    :type bead_radii: :class:`numpy.ndarray`

    :param n_structures: number of ensemble members
    :type n_structures: int
    
    :returns: a population of fully elongated structures
    :rtype: :class:`numpy.ndarray`
    """
    X = [bead_radii[0]]
    for i in range(len(bead_radii) -1):
        X.append(X[-1] + bead_radii[i+1] + bead_radii[i])
    X = np.array(X) - np.mean(X)
    
    X = np.array([X, np.zeros(len(bead_radii)),
                  np.zeros(len(bead_radii))]).T[None,:]
    X = X.repeat(n_structures, 0).ravel().astype(float)

    return X    

def make_random_structures(bead_radii, n_structures):
    """
    Makes a set of random structures with bead positions drawn
    from a normal distribution

    :param bead_radii: bead radii for each bead
    :type bead_radii: :class:`numpy.ndarray`

    :param n_structures: number of ensemble members
    :type n_structures: int

    :returns: a population of random structures
    :rtype: :class:`numpy.ndarray`
    """

    d = bead_radii.mean() * len(bead_radii) ** 0.333
    X = np.random.normal(scale=d, size=(n_structures, len(bead_radii), 3))
    
    return X.ravel()

def setup_initial_state(initial_state_params, posterior):
    """
    Sets up an initial state for MCMC sampling

    Depending on the settings in initial_state_params, the
    initial structures are either elongated and then slightly
    perturbed (currently disabled), random or loaded from a file

    :param initial_state_params: settings for initial state as
                                 specified in a config file
    :type initial_state_params: dict

    :param posterior: posterior distribution from which the
                      variables of the initial state are retreived
    :type posterior: :class:`binf.pdf.posteriors.Posterior`

    :returns: an state containing initial values for all variables
              of the posterior distribution
    :rtype: :class:`binf.samplers.BinfState`
    """
    from binf.samplers import BinfState

    p = posterior
    n_structures = p.likelihoods['ensemble_contacts'].forward_model.n_structures
    structures = initial_state_params['structures']
    norm = initial_state_params['norm']
    variables = p.variables

    if structures == 'elongated':
        bead_radii = posterior.priors['nonbonded_prior'].forcefield.bead_radii
        if False:
            structures = make_elongated_structures(bead_radii, n_structures)
            structures += np.random.normal(scale=0.5, size=structures.shape)
        else:            
            structures = make_random_structures(bead_radii, n_structures)
    else:
        try:
            structures = np.load(structures)
            if len(structures.shape) > 1:
                structures = structures[np.random.choice(structures.shape[0],
                                                         n_structures, replace=False)]
                structures = structures.ravel()
        except:
            raise ValueError('Couldn\'t load initial structures '
                             'from file {}!'.format(structures))

    init_state = BinfState({'structures': structures})

    if 'weights' in variables:
        raise NotImplementedError("Weights sampling not yet supported")
    if 'norm' in variables:
        init_state.update_variables(norm=norm)

    return init_state
    

def make_conditional_posterior(posterior, settings):
    """
    Conditions the posterior on the fixed variables

    :param posterior: full posterior distribution
    :type posterior: :class:`binf.pdf.posteriors.Posterior`

    :param settings: simulation settings as specified in a
                     config file
    :type settings: dict of dicts

    :returns: a copy of the input posterior distribution, but with
              some variables set to values specified in settings
    :rtype: :class:`binf.pdf.posteriors.Posterior`
    """
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
    

def make_backbone_prior(bead_radii, backbone_prior_params, n_beads,
                        n_structures):
    """
    Makes the default backbone prior object.

    :param bead_radii: list of bead radii
    :type bead_radii: :class:`numpy.ndarray`

    :param backbone_prior_params: settings for the backbone prior as
                                  specified in a config file
    :type backbone_prior_params: dict

    :param n_beads: number of beads in a single structure
    :type n_beads: int

    :param n_structures: number of ensemble members
    :type n_structures: int

    :returns: the backbone prior object with parameters set as given
              in the settings
    :rtype: :class:`.BackbonePrior`
    """
    from .backbone_prior import BackbonePrior

    if 'mol_ranges' in backbone_prior_params:
        mol_ranges = backbone_prior_params['mol_ranges']
    else:
        mol_ranges = None

    if mol_ranges is None:
        mol_ranges = np.array([0, n_beads])
    else:
        mol_ranges = np.loadtxt(mol_ranges).astype(int)
    bb_ll = [np.zeros(mol_ranges[i+1] - mol_ranges[i] - 1)
             for i in range(len(mol_ranges) - 1)]
    bb_ul = [np.array([bead_radii[j] + bead_radii[j+1]
              for j in range(mol_ranges[i], mol_ranges[i+1] - 1)])
             for i in range(len(mol_ranges) - 1)]  
    BBP = BackbonePrior('backbone_prior',
                        lower_limits=bb_ll, upper_limits=bb_ul,
                        k_bb=float(backbone_prior_params['force_constant']),
                        n_structures=n_structures,
                        mol_ranges=mol_ranges
                        )

    return BBP

def make_priors(nonbonded_prior_params, backbone_prior_params,
                sphere_prior_params, n_beads, n_structures):
    """
    Sets up all structural prior distributions

    :param nonbonded_prior_params: settings for the non-bonded prior
                                   as specified in a config file
    :type nonbonded_prior_params: dict

    :param backbone_prior_params: settings for the backbone prior
                                  as specified in a config file
    :type backbone_prior_params: dict

    :param sphere_prior_params: settings for a possible sphere prior
                                as specified in a config file
    :type sphere_prior_params: dict

    :param n_beads: number of beads in a single structure
    :type n_beads: int

    :param n_structures: number of ensemble members
    :type n_structures: int

    :returns: a dictionary with keys being the names of the
              structural prior distributions and values
              the prior objects themselves
    :rtype: dict of :class:`binf.pdf.priors.AbstractPrior`-derived
            objects
    """
    nb_params = nonbonded_prior_params
    
    try:
        bead_radius = float(nb_params['bead_radii'])
        bead_radii = np.ones(n_beads) * bead_radius
    except:
        bead_radii = np.loadtxt(nb_params['bead_radii'],
                                dtype=float)
        
    NBP = make_nonbonded_prior(nb_params, bead_radii, n_structures)
    BBP = make_backbone_prior(bead_radii, backbone_prior_params,
                              n_beads, n_structures)
    priors = {NBP.name: NBP, BBP.name: BBP}
    if sphere_prior_params['active'] == 'True':
        SP = make_sphere_prior(sphere_prior_params, bead_radii, n_structures)
        priors.update(**{SP.name: SP})

    return priors

def make_nonbonded_prior(nb_params, bead_radii, n_structures):
    """
    Makes the default non-bonded structural prior object.

    This will either be a Boltzmann-like distribution or a
    Tsallis ensemble (currently not really supported).

    :param nonbonded_prior_params: settings for the non-bonded prior as
                                   specified in a config file
    :type nonbonded_prior_params: dict

    :param bead_radii: list of bead radii
    :type bead_radii: :class:`numpy.ndarray`

    :param n_structures: number of ensemble members
    :type n_structures: int

    :returns: the non-bonded prior object with parameters set as given
              in the settings
    :rtype: :class:`.NonbondedPrior`
    """
    from .forcefields import ForceField
    from .forcefields import NBLForceField as ForceField

    forcefield = ForceField(bead_radii, float(nb_params['force_constant']))
    if not 'ensemble' in nb_params or nb_params['ensemble'] == 'boltzmann':
        from .nonbonded_prior import BoltzmannNonbondedPrior2    
        NBP = BoltzmannNonbondedPrior2('nonbonded_prior', forcefield,
                                       n_structures=n_structures, beta=1.0)
    elif nb_params['ensemble'] == 'tsallis':
        raise NotImplementedError
        from .nonbonded_prior import TsallisNonbondedPrior2
        NBP = TsallisNonbondedPrior('nonbonded_prior', bead_radii=bead_radii,
                                    force_constant=force_constant,
                                    n_structures=n_structures, q=1.0)

    return NBP

def make_sphere_prior(sphere_prior_params, bead_radii, n_structures):
    """
    Makes a sphere structural prior object.

    This is a Boltzmann-like distribution with a potential energy
    harmonically restraining all beads to stay within a sphere
    of a given radius.

    :param sphere_prior_params: settings for the sphere prior as
                                specified in a config file
    :type sphere_prior_params: dict

    :param bead_radii: list of bead radii
    :type bead_radii: :class:`numpy.ndarray`

    :param n_structures: number of ensemble members
    :type n_structures: int

    :returns: the sphere prior object with parameters set as given
              in the settings
    :rtype: :class:`.SpherePrior`
    """
    from .sphere_prior import SpherePrior
    radius = sphere_prior_params['radius']
    if radius == 'auto':
        radius = 2 * bead_radii.mean() * len(bead_radii) ** (1 / 3.0)
    else:
        radius = float(radius)
    SP = SpherePrior('sphere_prior',
                     sphere_radius=radius,
                     sphere_k=float(sphere_prior_params['force_constant']),
                     n_structures=n_structures, bead_radii=bead_radii)

    return SP    

def parse_data(data_file, data_filtering_params):
    
    disregard_lowest = data_filtering_params['disregard_lowest']
    ignore_seq_nbs = int(data_filtering_params['ignore_sequential_neighbors'])
    include_zero_counts = data_filtering_params['include_zero_counts']
    data = np.loadtxt(data_file, dtype=int)
    if include_zero_counts == 'False':
        data = data[data[:,2] > 0]
    data = data[np.argsort(data[:,2])]
    data = data[int(disregard_lowest * len(data)):]
    data = data[np.abs(data[:,0] - data[:,1]) > ignore_seq_nbs]

    return data
    
def make_likelihood(forward_model_params, error_model, data_filtering_params,
                    data_file, n_structures, bead_radii):
    """
    Sets up a likelihood object from settings parsed from a config
    file

    :param forward_model_params: settings for the forward model as
                                 specified in a config file
    :type forward_model_params: dict

    :param error_model: a string telling which error model to use.
                        This is gonna be either 'poisson', 'lognormal',
                        or 'gaussian'. At the moment, only 'poisson'
                        is supported, but code for others is provided.
    :type error_model: string

    :param data_filtering_params: settings for filtering data as
                                  specified in a config file
    :type data_filtering_params: dict

    :param data_file: path to the text file containing the list of
                      pairwise bead contact frequencies
    :type data_file: string

    :param n_structures: number of ensemble members
    :type n_structures: int

    :param bead_radii: list of bead radii
    :type bead_radii: :class:`numpy.ndarray`

    :returns: the ready-to-use likelihood object
    :rtype: :class:`.Likelihood`
    """

    from .forward_models import EnsembleContactsFWM
    from .likelihoods import Likelihood

    data = parse_data(data_file, data_filtering_params)
    cd_factor = float(forward_model_params['contact_distance_factor'])
    contact_distances = (bead_radii[data[:,0]] + bead_radii[data[:,1]]) * cd_factor
        
    FWM = EnsembleContactsFWM('fwm', n_structures, contact_distances,
                              data_points=data)

    if error_model == 'poisson':
        from .error_models import PoissonEM
        EM = PoissonEM('ensemble_contacts_em', data[:,2])
    else:
        raise(NotImplementedError)

    L = Likelihood('ensemble_contacts', FWM, EM, 1.0)
    L = L.conditional_factory(smooth_steepness=forward_model_params['alpha'])
    
    return L

def setup_default_re_master(n_replicas, sim_path, comm):
    """
    Sets up a Replica Exchange master object from the :class:`rexfw` package

    This object manages a Replica Exchange simulation, that is, it tells
    :class:`rexfw.replicas.Replica`s when to sample, when to exchange states,
    when to dump states, do send statistics etc.
    It differs from the re

    :param n_replicas: number of replicas. Using MPI, for N replicas,
                       you need N+1 processes (one process is required for
                       the Replica Exchange master object)
    :type n_replicas: int

    :param sim_path: folder where all simulation output will be stored.
                     You'll want this to be on the scratch space of your
                     HPC systems, as all the samples (and statistic files)
                     will be written there
    :type sim_path: string

    :param comm: a communicator object which is required by many :class:`rexfw`
                 objects to comunnicate with replicas and the master object.
                 Currently, only a MPI-based communicator is implemented.
    :type comm: :class:`rexfw.communicators.AbstractCommunicator`, most likely
                :class:`rexfw.communicators.mpi.MPICommunicator`

    :returns: a fully set-up Replica Exchange master object
    :rtype: :class:`rexfw.remaster.ExchangeMaster`
    """    

    from rexfw.remasters import ExchangeMaster
    from rexfw.statistics import Statistics, REStatistics
    from rexfw.statistics.writers import StandardConsoleREStatisticsWriter, StandardFileMCMCStatisticsWriter, StandardFileREStatisticsWriter, StandardFileREWorksStatisticsWriter, StandardConsoleMCMCStatisticsWriter, StandardConsoleMCMCStatisticsWriter
    from rexfw.convenience import create_default_RE_params
    from rexfw.convenience.statistics import create_default_RE_averages, create_default_MCMC_averages, create_default_works, create_default_stepsizes, create_default_heats

    replica_names = ['replica{}'.format(i) for i in range(1, n_replicas + 1)]
    params = create_default_RE_params(n_replicas)
        
    local_pacc_avgs = create_default_MCMC_averages(replica_names, 'structures')
    re_pacc_avgs = create_default_RE_averages(replica_names)
    stepsizes = create_default_stepsizes(replica_names, 'structures')

    works = create_default_works(replica_names)
    heats = create_default_heats(replica_names)
    stats_path = sim_path + 'statistics/'
    works_path = sim_path + 'works/'

    import os
    for p in (stats_path, works_path):
        if not os.path.exists(p):
            os.makedirs(p)

    stats_writers = [StandardConsoleMCMCStatisticsWriter(['structures'],
                                                         ['acceptance rate',
                                                          'stepsize']),
                     StandardFileMCMCStatisticsWriter(stats_path + 'mcmc_stats.txt',
                                                      ['structures'],
                                                      ['acceptance rate', 'stepsize'])
                    ]
    stats = Statistics(elements=local_pacc_avgs + stepsizes, 
                       stats_writer=stats_writers)
    re_stats_writers = [StandardConsoleREStatisticsWriter(),
                        StandardFileREStatisticsWriter(stats_path + 're_stats.txt',
                                                       ['acceptance rate'])]
    works_writers = [StandardFileREWorksStatisticsWriter(works_path)]
    re_stats = REStatistics(elements=re_pacc_avgs,
                            work_elements=works, heat_elements=heats,
                            stats_writer=re_stats_writers,
                            works_writer=works_writers)
    
    master = ExchangeMaster('master0', replica_names, params, comm=comm, 
                            sampling_statistics=stats, swap_statistics=re_stats)

    return master
