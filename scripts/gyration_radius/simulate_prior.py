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
    rank = 4
    size = 5
    config_file = '/usr/users/scarste/projects/ensemble_hic/scripts/bau2011/tmpcfg.cfg'
n_replicas = size - 1
settings = parse_config_file(config_file)

comm = MPICommunicator()

re_params = settings['replica']
bead_radii = np.loadtxt(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/bead_radii.txt'))
n_beads = len(bead_radii)
betas = expspace(0, 200, 
schedule = dict(target_rog=target_rogs)

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
    from ensemble_hic.nonbonded_prior import BoltzmannNonbondedPrior2
    from ensemble_hic.forcefields import NBLForceField
    from ensemble_hic.hmc import FastHMCSampler
    
    BBP = BackbonePrior('backbone_prior',
                        np.zeros(len(bead_radii) - 1)[None,:],
                        (bead_radii[:-1] + bead_radii[1:])[None,:],
                        500,
                        1)

    from isd2.pdf import AbstractISDPDF
    class MaxEntROGPotential(AbstractISDPDF):

        def __init__(self, delta, bead_radii):

            from csb.statistics.pdf.parameterized import Parameter
            from isd2 import ArrayParameter

            super(MaxEntROGPotential, self).__init__('max_ent_rog_potential')

            self.bead_radii = bead_radii
            self._register_variable('structures', differentiable=True)
            self._register('delta')
            self['delta'] = Parameter(delta, 'delta')        
            self.update_var_param_types(structures=ArrayParameter)
            self._set_original_variables()

        def _wmean(self, X, w):

            return np.sum(X * w[:,None], 0) / X.shape[0]
            
        def _calculate_wrog_squared(self, X):

            sqrt = np.sqrt
            sum = np.sum
            n_beads = X.shape[0]
            weights = n_beads * self.bead_radii ** 3 / np.sum(self.bead_radii ** 3)

            return sum(weights * np.sum((X - self._wmean(X, weights)) ** 2, 1))
        
        def _evaluate_log_prob(self, structures):

            X = structures.reshape(-1, 3)

            return -self['delta'].value * np.sqrt(self._calculate_wrog_squared(X))

        def gradient(self, structures):

            sqrt = np.sqrt
            sum = np.sum
            X = structures.reshape(-1, 3)
            n_beads = X.shape[0]
            delta = self['delta'].value
            weights = n_beads * self.bead_radii ** 3 / np.sum(self.bead_radii ** 3)
            wrog = np.sqrt(self._calculate_wrog_squared(X))

            b = delta / wrog * (weights[:,None] * (X - self._wmean(X, weights)))

            return b.ravel()

        def clone(self):

            copy = self.__class__(delta=self['delta'].value,
                                  bead_radii=self.bead_radii)

            copy.set_fixed_variables_from_pdf(self)

            return copy

            
    
    FF = NBLForceField(bead_radii, 50)
    NBP = BoltzmannNonbondedPrior2('nonbonded_prior', FF, 1, 1.0)
    ROGP = MaxEntROGPotential(schedule['beta'][rank-1], FF.bead_radii)
    posterior = Posterior({}, {'backbone_prior': BBP,
                               'rog_potential': ROGP,
                               'nonbonded_prior': NBP})

    initial_state = ISDState({'structures': np.random.uniform(-np.sum(bead_radii),
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
