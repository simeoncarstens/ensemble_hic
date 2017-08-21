import numpy, os, sys
this_path = '~/projects/hic/py/hicisd2/ensemble_scripts/protein_test/'
this_path = os.path.expanduser(this_path)
sys.path.append(this_path)
from mpi4py import MPI

from rexfw.communicators.mpi import MPICommunicator
from rexfw.convenience import create_standard_RE_params, create_directories
from cPickle import dump
np = numpy

simname = sys.argv[1]
n_structures = int(sys.argv[2])
variables = sys.argv[3]
em = sys.argv[4]
prot1 = sys.argv[5]
prot2 = sys.argv[6]

mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
size = mpicomm.Get_size()

n_replicas = size - 1

ens_size = 200
smooth_steepness = 14.0
data_file = this_path + '{}_{}_maxwell_poisson_data_es100_sigma0.05.txt'.format(prot1, prot2)

outpath = '/scratch/scarste/ensemble_hic/{}_{}_{}_{}structures_{}replicas/'.format(simname, em, variables, n_structures, n_replicas)
# outpath = '/tmp/ensemble_hic/{}_{}replicas/'.format(simname, n_replicas)
create_directories(outpath)

if False:
    ## copy this file and current REXFW version to output directory
    srcdir = outpath+'src/'
    os.system('mkdir '+srcdir)
    import inspect
    os.system('cp {} {}'.format(inspect.getfile(inspect.currentframe()), srcdir))
    os.system('cp -r {} {}'.format(os.path.expanduser('~/projects/rexfw/'),
                                   srcdir))

comm = MPICommunicator()

from stuff import expspace
schedule = {'lammda': np.linspace(0.001, 1, n_replicas),
            'beta': np.ones(n_replicas)}
# schedule = {'lammda': np.concatenate((np.zeros(n_replicas / 2) + 0.001,
#                                       np.linspace(0.001, 1, n_replicas / 2))),
#             'beta': np.concatenate((np.linspace(0.001, 1, n_replicas / 2),
#                                     np.ones(n_replicas / 2)))}

if rank == 0:

    from misc import setup_default_re_master

    with open(outpath + 'schedule.pickle','w') as opf:
        dump(schedule, opf)

    master = setup_default_re_master(n_replicas, outpath, comm)

    dump_interval = 100
    n_samples = 20001

    master.run(n_samples, swap_interval=5, status_interval=50,
               dump_interval=dump_interval,
               samples_folder=outpath + 'samples/', dump_step=20,
               statistics_update_interval=40)
    master.terminate_replicas()

else:
    
    from csb.bio.io import StructureParser
    from rexfw.replicas import Replica
    from rexfw.slaves import Slave
    from rexfw.proposers import REProposer
    from isd2.samplers.gibbs import GibbsSampler
    from isd2.samplers import ISDState

    coords  = StructureParser(prot1 + '.pdb').parse().get_coordinates(['CA'])
    n_beads = len(coords)
    
    X = np.random.normal(size=n_beads * n_structures * 3, scale=5)
    X = np.array([range(0, 2 * n_beads, 2),
                  np.zeros(n_beads),
                  np.zeros(n_beads)]).T[None,:]
    
    X = X.repeat(n_structures, 0).ravel().astype(float)
    X += np.random.normal(scale=0.1, size=X.shape)

    if em == 'gaussian':
        from misc import make_posterior
    elif em == 'poisson':
        from misc import make_poisson_posterior
        make_posterior = make_poisson_posterior
    elif em == 'lognormal':
        from misc import make_lognormal_posterior
        make_posterior = make_lognormal_posterior
        
    posterior = make_posterior(n_structures,
                               data_file, 
                               # this_path + '{}_{}_maxwell_poisson_data_es100.txt'.format(prot1, prot2),
                               # this_path + 'fwm_created_data_23beads.txt',
                               n_beads=n_beads,
                               smooth_steepness=smooth_steepness,
                               contact_distance=1.8,
                               disregard_lowest=0.0,
                               beta=schedule['beta'][rank-1],
                               lammda=schedule['lammda'][rank-1],
                               k_ve=50.0, k2=100.0, k_bb=1000.0,
                               ignore_sequential_neighbors=1,
                               include_zero_counts=True,
                               gradient_cutoff=10000.0)

    if variables == 'sw':
        from misc import make_subsamplers_weights
        posterior = posterior.conditional_factory(norm=1.0)
        if em in ('gaussian', 'lognormal'):
            posterior = posterior.conditional_factory(k2=100.0)
        weights = np.ones(n_structures) * ens_size / float(n_structures)
        state = ISDState({'structures': X,
                          'weights': weights,
                          #'k2': 1.0
                          })
        subsamplers = make_subsamplers_weights(posterior, state,
                                               structures_timestep=1e-4,
                                               structures_nsteps=100,
                                               weights_timestep=1e-1,
                                               weights_nsteps=50)
    elif variables == 'sn':
        from misc import make_subsamplers_norm
        posterior = posterior.conditional_factory(weights=np.ones(n_structures))
        if em in ('gaussian', 'lognormal'):
            posterior = posterior.conditional_factory(k2=5e-4)
        norm = ens_size / float(n_structures)
        state = ISDState({'structures': X, 'norm': norm})
        subsamplers = make_subsamplers_norm(posterior, state,
                                            structures_timestep=1e-4,
                                            structures_nsteps=100,
                                            norm_stepsize=None)
    elif variables == 's':
        from misc import make_subsamplers_onlystructures

        posterior = posterior.conditional_factory(weights=np.ones(n_structures),
                                                  norm=ens_size/float(n_structures))
        if em in ('gaussian', 'lognormal'):
            posterior = posterior.conditional_factory(k2=100.0)
        state = ISDState({'structures': X})
        subsamplers = make_subsamplers_onlystructures(posterior, state,
                                                      structures_timestep=1e-4,
                                                      structures_nsteps=100)
    elif variables == 'sk2':
        from misc import make_subsamplers_onlystructures
        posterior = posterior.conditional_factory(weights=np.ones(n_structures),
                                                  norm=ens_size/float(n_structures))
        state = ISDState({'structures': X,
                          'k2': 10.0
                          })
        subsamplers = make_subsamplers_onlystructures(posterior, state,
                                                      structures_timestep=1e-4,
                                                      structures_nsteps=100)
        

    if False:
        ## create an optimal state
        from misc import load_pdb
        from isd2.samplers import ISDState
        X1 = load_pdb('snake.pdb')
        X2 = load_pdb('hairpin.pdb')
        sigma = 0.05
        X = numpy.array([X1 + 0.0*numpy.random.normal(size=X1.shape, scale=sigma) for _ in range(n_structures / 2)])
        X = numpy.vstack((X, [X2 + 0.0*numpy.random.normal(size=X2.shape, scale=sigma) for _ in range(n_structures / 2)]))
        vardict = dict(structures=X)
        if variables == 'sw':
            weights = numpy.ones(n_structures) * ens_size / float(n_structures)
            vardict.update(weights=weights)
        elif variables == 'sn':
            norm = exp_ensemble_size / float(n_structures)
            vardict.update(norm=norm)
        state = ISDState(vardict)

    sampler = GibbsSampler(pdf=posterior, state=state,
                           subsamplers=subsamplers)    
    proposer = REProposer('prop{}'.format(rank))
    proposers = {proposer.name: proposer}
    replica = Replica('replica{}'.format(rank), state, 
                      posterior, {},
                      GibbsSampler,
                      {'subsamplers': subsamplers},
                      proposers, comm)

    slave = Slave({'replica{}'.format(rank): replica}, comm)

    slave.listen()
        
