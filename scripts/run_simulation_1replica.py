import os
import sys
import numpy as np
import ConfigParser

from ensemble_hic.setup_functions import make_replica_schedule, parse_config_file

if False:
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
    config_file = ppath + 'scripts/eser2017/tmpcfg_wholegenome.cfg'
    config_file = ppath + 'scripts/proteins/test.cfg'
    config_file = ppath + 'scripts/proteins/mergetest.cfg'
    settings = parse_config_file(config_file)
    # settings['general']['data_file'] = ppath + 'data/rao2014/chr1.txt'
    # settings['nonbonded_prior']['bead_radii'] = ppath + 'data/rao2014/chr1_radii.txt'
    #settings['general']['n_structures'] = '3'

n_replicas = size - 1

re_params = settings['replica']
if not re_params['schedule'] in ('linear', 'exponential'):
    schedule = np.load(re_params['schedule'])
else:
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
initial_state.update_variables(norm=1.0)
subsamplers = make_subsamplers(posterior, initial_state.variables,
                               settings['structures_hmc'],
                               settings['weights_hmc'])

sampler = GibbsSampler(pdf=posterior, state=initial_state,
                       subsamplers=subsamplers)    
