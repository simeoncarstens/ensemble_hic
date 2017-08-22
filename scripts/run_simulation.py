import os
import sys
import numpy as np
from mpi4py import MPI

from rexfw.communicators.mpi import MPICommunicator
from rexfw.convenience import create_standard_RE_params, create_directories
from cPickle import dump

from ensemble_hic.setup_functions import make_replica_schedule

mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
size = mpicomm.Get_size()
n_replicas = size - 1

import ConfigParser
config = ConfigParser.ConfigParser()
config.read(sys.argv[1])

def config_section_map(section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
        except:
            dict1[option] = None
    return dict1

general_params = config_section_map('general')
forward_model_params = config_section_map('forward_model')
sphere_prior_params = config_section_map('sphere_prior')
nonbonded_prior_params = config_section_map('nonbonded_prior')
backbone_prior_params = config_section_map('backbone_prior')
data_filtering_params = config_section_map('data_filtering')
initial_state_params = config_section_map('initial_state')
replica_params = config_section_map('replica')
structures_hmc_params = config_section_map('structures_hmc')
weights_hmc_params = config_section_map('weights_hmc')

comm = MPICommunicator()

schedule = make_replica_schedule(replica_params, n_replicas)

if rank == 0:

    from ensemble_hic.setup_functions import setup_default_re_master
    from rexfw.convenience import create_directories
    from shutil import copy2

    output_folder = general_params['output_folder']
    if output_folder[-1] != '/':
        output_folder += '/'
    create_directories(output_folder)
    copy2(sys.argv[1], output_folder + 'config.cfg')
    with open(output_folder + 'schedule.pickle','w') as opf:
        dump(schedule, opf)

    ## setup replica exchange
    master = setup_default_re_master(n_replicas, output_folder, comm)

    ## run replica exchange
    master.run(int(replica_params['n_samples']) + 1,
               swap_interval=int(replica_params['swap_interval']),
               status_interval=int(replica_params['print_status_interval']),
               dump_interval=int(replica_params['samples_dump_interval']),
               samples_folder=output_folder + 'samples/',
               dump_step=int(replica_params['samples_dump_step']),
               statistics_update_interval=int(replica_params['statistics_update_interval']))

    ## kill replicas
    master.terminate_replicas()

else:
    
    from rexfw.replicas import Replica
    from rexfw.slaves import Slave
    from rexfw.proposers import REProposer

    from isd2.samplers.gibbs import GibbsSampler
    from isd2.pdf.posteriors import Posterior
    
    from ensemble_hic.setup_functions import make_priors, make_likelihood
    from ensemble_hic.setup_functions import make_conditional_posterior, make_subsamplers
    from ensemble_hic.setup_functions import setup_initial_state, setup_weights
    

    lammda = schedule['lammda'][rank - 1]
    beta   = schedule['beta'][rank - 1]
    n_structures = int(config.get('general', 'n_structures'))
    n_beads = int(config.get('general', 'n_beads'))
    data_file = config.get('general', 'data_file')
    
    priors = make_priors(nonbonded_prior_params,
                         backbone_prior_params,
                         sphere_prior_params,
                         n_beads, n_structures)
    bead_radii = priors['nonbonded_prior'].bead_radii
    likelihood = make_likelihood(forward_model_params,
                                 general_params['error_model'],
                                 data_filtering_params,
                                 data_file, n_structures, bead_radii)
    posterior = Posterior({likelihood.name: likelihood}, priors)
    posterior['lammda'].set(lammda)
    posterior['beta'].set(beta)

    variables = general_params['variables']
    initial_state_params['weights'] = setup_weights(initial_state_params,
                                                    n_structures)
    posterior = make_conditional_posterior(posterior, initial_state_params, variables)

    initial_state = setup_initial_state(initial_state_params, posterior)
    subsamplers = make_subsamplers(posterior, initial_state.variables,
                                   structures_hmc_params, weights_hmc_params)

    sampler = GibbsSampler(pdf=posterior, state=initial_state,
                           subsamplers=subsamplers)    
    proposer = REProposer('prop{}'.format(rank))
    proposers = {proposer.name: proposer}
    replica = Replica('replica{}'.format(rank), initial_state, 
                      posterior, {},
                      GibbsSampler,
                      {'subsamplers': subsamplers},
                      proposers, comm)

    slave = Slave({'replica{}'.format(rank): replica}, comm)

    slave.listen()
        

if not True:
    import sys
    sys.argv[1]='bau2011/config.cfg'
    size=10
    rank=9
    n_replicas = size-1

    L=posterior.likelihoods['ensemble_contacts']
    fwm=L.forward_model

    
