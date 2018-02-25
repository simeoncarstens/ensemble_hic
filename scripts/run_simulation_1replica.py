import os
import sys
import numpy as np
import ConfigParser

from ensemble_hic.setup_functions import make_replica_schedule, parse_config_file

if True:
    np.random.seed(42)
    rank = 2
    size = 3
    config_file = '/home/simeoncarstens/projects/ensemble_hic/scripts/rao2014/tmpcfg.cfg'
if True:
    np.random.seed(42)
    rank = 2
    size = 3
    config_file = '/home/simeon/projects/ensemble_hic/scripts/rao2014/tmpcfg.cfg'

n_replicas = size - 1
settings = parse_config_file(config_file)

re_params = settings['replica']
schedule = make_replica_schedule(re_params, n_replicas)
    
from isd2.samplers.gibbs import GibbsSampler

from ensemble_hic.setup_functions import make_posterior, make_subsamplers
from ensemble_hic.setup_functions import setup_initial_state, setup_weights

settings['initial_state']['weights'] = setup_weights(settings)    
posterior = make_posterior(settings)
for replica_parameter in schedule:
    posterior[replica_parameter].set(schedule[replica_parameter][rank - 1])

initial_state = setup_initial_state(settings['initial_state'], posterior)
if not 'norm' in initial_state.variables:
    posterior['norm'].set(np.max(posterior.likelihoods['ensemble_contacts'].error_model.data) / float(settings['general']['n_structures']))
subsamplers = make_subsamplers(posterior, initial_state.variables,
                               settings['structures_hmc'],
                               settings['weights_hmc'])

sampler = GibbsSampler(pdf=posterior, state=initial_state,
                       subsamplers=subsamplers)    
