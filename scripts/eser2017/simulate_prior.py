import os
import sys
import numpy as np
import ConfigParser
from mpi4py import MPI

from rexfw.communicators.mpi import MPICommunicator
from rexfw.convenience import create_standard_RE_params, create_directories
from cPickle import dump

from ensemble_hic.setup_functions import make_replica_schedule, parse_config_file
from ensemble_hic.setup_functions import expspace

mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
size = mpicomm.Get_size()
config_file = sys.argv[1]

if not True:
    np.random.seed(42)
    rank = 3
    size = 5
    config_file = '/usr/users/scarste/projects/ensemble_hic/scripts/eser2017/tmpcfg_wholegenome_prior.cfg'
n_replicas = size - 1
settings = parse_config_file(config_file)

comm = MPICommunicator()

re_params = settings['replica']
betas = expspace(0, 1, 0.015, n_replicas)
schedule = dict(beta=betas)

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

    from isd2.samplers import BinfState
    from isd2.samplers.gibbs import GibbsSampler
    from isd2.pdf.posteriors import Posterior

    from ensemble_hic.hmc import FastHMCSampler
    from ensemble_hic.setup_functions import make_sphere_prior, make_nonbonded_prior
    from ensemble_hic.setup_functions import make_backbone_prior
    
    bead_radii = np.loadtxt(settings['nonbonded_prior']['bead_radii'])
    n_beads = len(bead_radii)
    mol_ranges = np.loadtxt(settings['backbone_prior']['mol_ranges'], dtype=int)
    SP = make_sphere_prior(settings['sphere_prior'], bead_radii, 1)
    NBP = make_nonbonded_prior(settings['nonbonded_prior'], bead_radii, 1)
    NBP['beta'].set(betas[rank-1])
    BBP = make_backbone_prior(bead_radii, settings['backbone_prior'], n_beads, 1)
    posterior = Posterior({}, {'backbone_prior': BBP,
                               'sphere_prior': SP,
                               'nonbonded_prior': NBP})

    initial_state = BinfState({'structures': np.random.uniform(-np.sum(bead_radii),
                                                              np.sum(bead_radii),
                                                              n_beads * 3)})

    timestep = float(settings['structures_hmc']['timestep'])
    n_steps = int(settings['structures_hmc']['trajectory_length'])
    adaption_limit = int(settings['structures_hmc']['adaption_limit'])
    
    subsamplers = {'structures': FastHMCSampler(posterior,
                                                initial_state.variables['structures'],
                                                timestep,
                                                n_steps,
                                                adaption_limit,
                                                variable_name='structures')}
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
