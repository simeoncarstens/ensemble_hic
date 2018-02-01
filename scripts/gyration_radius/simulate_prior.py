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
config_file = sys.argv[1]
if True:
    np.random.seed(42)
    rank = 4
    size = 5
    config_file = '/usr/users/scarste/projects/ensemble_hic/scripts/bau2011/tmpcfg.cfg'
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

    from isd2.samplers import ISDState
    from isd2.samplers.gibbs import GibbsSampler
    from isd2.pdf.posteriors import Posterior

    from ensemble_hic.backbone_prior import BackbonePrior
    from ensemble_hic.rog_prior import GyrationRadiusPrior
    from ensemble_hic.nonbonded_prior import BoltzmannNonbondedPrior2
    from ensemble_hic.forcefields import NBLForceField
    from ensemble_hic.hmc import FastHMCSampler
    
    bead_radii = np.loadtxt(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/bead_radii.txt'))
    n_beads = len(bead_radii)
    target_rogs = np.linspace(10.0 * bead_radii.sum() ** 0.3333,
                              0.1 * bead_radii.sum() ** 0.3333,
                              n_replicas)
    BBP = BackbonePrior('backbone_prior',
                        np.zeros(len(bead_radii) - 1)[None,:],
                        (bead_radii[:-1] + bead_radii[1:])[None,:],
                        500,
                        1)
    ROGP = GyrationRadiusPrior('rog_prior', target_rogs[rank - 1], 5.0, 1)
    FF = NBLForceField(bead_radii, 50)
    NBP = BoltzmannNonbondedPrior2('nonbonded_prior', FF, 1, 1.0)
    posterior = Posterior({}, {'backbone_prior': BBP,
                               'rog_prior': ROGP,
                               'nonbonded_prior': NBP})

    initial_state = ISDState({'structures': np.random.uniform(-np.sum(bead_radii),
                                                              np.sum(bead_radii),
                                                              n_beads * 3)})

    subsamplers = {'structures': FastHMCSampler(posterior,
                                                initial_state.variables['structures'],
                                                1e-2, 100, 1000000)}
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
