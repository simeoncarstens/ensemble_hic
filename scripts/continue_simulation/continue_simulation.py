import os
import sys
import numpy as np
import ConfigParser
from mpi4py import MPI

from rexfw.communicators.mpi import MPICommunicator
from rexfw.convenience import create_directories
from cPickle import dump

from ensemble_hic.setup_functions import make_replica_schedule, parse_config_file

mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
size = mpicomm.Get_size()
config_file = sys.argv[1]
n_replicas = size - 1
settings = parse_config_file(config_file)

comm = MPICommunicator()

re_params = settings['replica']
if re_params['schedule'] in ('linear', 'exponential'):
    schedule = make_replica_schedule(re_params, n_replicas)
elif re_params['schedule'][-3:] == '.py':
    # exec(open(re_params['schedule']).read())
    import numpy as np
    from scipy import stats
    from mpi4py import MPI
    space = np.linspace(0, 1, n_replicas)
    m = float(re_params['gauss_mean'])#.65
    s = float(re_params['gauss_std'])#0.2
    delta_betas = stats.norm.pdf(space, m, s)
    delta_betas = [0] + delta_betas
    betas = np.cumsum(delta_betas)
    betas /= betas[-1]

    schedule = {'lammda': betas, 'beta': betas}

else:
    schedule = np.load(re_params['schedule'])

cont_folder = settings['general']['cont_folder']

output_folder = settings['general']['output_folder']
if output_folder[-1] != '/':
    output_folder += '/'

if rank == 0:

    from ensemble_hic.setup_functions import setup_default_re_master
    from rexfw.convenience import create_directories
    from shutil import copy2

    ## Setup replica exchange
    master = setup_default_re_master(n_replicas, cont_folder, comm)

    ## run replica exchange
    offset = int(settings['replica']['offset'])
    master.run(int(re_params['n_samples']) + 1,
               swap_interval=int(re_params['swap_interval']),
               status_interval=int(re_params['print_status_interval']),
               dump_interval=int(re_params['samples_dump_interval']),
               dump_step=int(re_params['samples_dump_step']),
               statistics_update_interval=int(re_params['statistics_update_interval']),
               offset=offset)

    ## kill replicas
    master.terminate_replicas()

else:
    
    from rexfw.replicas import Replica
    from rexfw.slaves import Slave
    from rexfw.proposers.re import REProposer

    from isd2.samplers.gibbs import GibbsSampler
    
    from ensemble_hic.setup_functions import make_posterior, make_subsamplers
    from ensemble_hic.setup_functions import setup_initial_state
    from ensemble_hic.replica import CompatibleReplica

    posterior = make_posterior(settings)
    for replica_parameter in schedule:
        posterior[replica_parameter].set(schedule[replica_parameter][rank - 1])

    initial_state = setup_initial_state(settings['initial_state'], posterior)
    if not 'norm' in initial_state.variables:
        posterior['norm'].set(np.max(posterior.likelihoods['ensemble_contacts'].error_model.data) / float(settings['general']['n_structures']))
    initial_state.update_variables(structures=np.load(cont_folder + 'init_states.npy')[rank - 1])
    initial_state.update_variables(norm=np.load(cont_folder + 'init_norms.npy')[rank - 1])
    settings['structures_hmc'].update(timestep=np.load(cont_folder + 'timesteps.npy')[rank - 1])
    subsamplers = make_subsamplers(posterior, initial_state.variables,
                                   settings['structures_hmc'])

    sampler = GibbsSampler(pdf=posterior, state=initial_state,
                           subsamplers=subsamplers)    
    proposer = REProposer('prop{}'.format(rank))
    proposers = {proposer.name: proposer}
    replica = CompatibleReplica('replica{}'.format(rank), initial_state, 
                                posterior,
                                GibbsSampler,
                                {'subsamplers': subsamplers},
                                proposers, output_folder, comm)

    slave = Slave({'replica{}'.format(rank): replica}, comm)

    slave.listen()
