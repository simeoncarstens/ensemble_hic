import os
import sys
import numpy as np
import ConfigParser
from mpi4py import MPI

from rexfw.communicators.mpi import MPICommunicator
from rexfw.convenience import create_standard_RE_params, create_directories
from cPickle import dump

from ensemble_hic.setup_functions import make_replica_schedule, parse_config_file

mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
size = mpicomm.Get_size()
n_replicas = size - 1
config_file = sys.argv[1]
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

if rank == 0:

    from ensemble_hic.setup_functions import setup_default_re_master
    from rexfw.convenience import create_directories
    from shutil import copy2

    output_folder = settings['general']['output_folder']
    if output_folder[-1] != '/':
        output_folder += '/'
    create_directories(output_folder)
    copy2(config_file, output_folder + 'config.cfg')
    with open(output_folder + 'schedule.pickle','w') as opf:
        dump(schedule, opf)

    ## setup replica exchange
    master = setup_default_re_master(n_replicas, output_folder, comm)

    ## run replica exchange
    master.run(int(re_params['n_samples']) + 1,
               swap_interval=int(re_params['swap_interval']),
               status_interval=int(re_params['print_status_interval']),
               dump_interval=int(re_params['samples_dump_interval']),
               samples_folder=output_folder + 'samples/',
               dump_step=int(re_params['samples_dump_step']),
               statistics_update_interval=int(re_params['statistics_update_interval']))

    ## kill replicas
    master.terminate_replicas()

else:
    
    from rexfw.replicas import Replica
    from rexfw.slaves import Slave
    from rexfw.proposers import REProposer

    from isd2.samplers.gibbs import GibbsSampler
    
    from ensemble_hic.setup_functions import make_marginalized_posterior, make_posterior
    from ensemble_hic.setup_functions import setup_initial_state, setup_weights
    from ensemble_hic.setup_functions import make_elongated_structures
    from isd2.samplers.hmc import ISD2HMCSampler

    oldp = make_posterior(settings)
    posterior = make_marginalized_posterior(settings)
    for replica_parameter in schedule:
        posterior[replica_parameter].set(schedule[replica_parameter][rank - 1])

    bead_radii = oldp.priors['nonbonded_prior'].bead_radii
    n_structures = oldp.priors['nonbonded_prior'].n_structures
    structures = make_elongated_structures(bead_radii, n_structures)
    structures += np.random.normal(scale=0.5, size=structures.shape)
    from isd2.samplers import ISDState
    initial_state = ISDState({'structures': structures})
    structures_hmc_params = settings['structures_hmc']
    
    structures_tl = int(structures_hmc_params['trajectory_length'])
    structures_timestep = float(structures_hmc_params['timestep'])
    structures_adaption = structures_hmc_params['timestep_adaption']
    structures_adaption = True if structures_adaption == 'True' else False
    if not structures_adaption:
        raise NotImplementedError('At the moment, timestep cannot be switched off!')
    from csb.statistics.samplers import State
    structures_sampler = ISD2HMCSampler(posterior,
                                        initial_state.variables['structures'],
                                        structures_timestep, structures_tl,
                                        variable_name='structures')
    sampler = structures_sampler
    subsamplers = {'structures': sampler}
    sampler = GibbsSampler(pdf=posterior, state=initial_state,
                           subsamplers=subsamplers)
    proposer = REProposer('prop{}'.format(rank))
    proposers = {proposer.name: proposer}
    replica = Replica('replica{}'.format(rank),
                      initial_state,
                      posterior, {},
                      GibbsSampler,
                      {'subsamplers': subsamplers},
                      proposers, comm)

    slave = Slave({'replica{}'.format(rank): replica}, comm)

    slave.listen()
