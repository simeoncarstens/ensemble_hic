import numpy, os
os.chdir(os.path.expanduser('~/projects/hic/py/hicisd2/ensemble_scripts/bau5C_test/'))
import matplotlib.pyplot as plt
from misc import make_poisson_posterior
from csb.bio.io import StructureParser

dataset = 'K562'
dataset = 'GM12878'

n_structures = 1
n_replicas = 40
variables = 'sn'
em = 'poisson'
smooth_steepness = 10.0
sim_name = 'bau5C_isn2_3rdrootradii_smallersphere_{}_{}_{}'.format(dataset, em, variables)

sim_path = '/scratch/scarste/ensemble_hic/{}_{}structures_{}replicas/'.format(sim_name, n_structures, n_replicas)
schedule = numpy.load(sim_path + 'schedule.pickle')['lammda']

data_file = os.path.expanduser('~/projects/hic/data/bau2011/{}.txt'.format(dataset))
n_beads = 70
from misc import make_data
data, _ = make_data(data_file)
ens_size = max(data[:,0])

from hicisd2.hicisd2lib import load_samples
samples = load_samples(sim_path + 'samples/', n_replicas, 38601, 100, burnin=5000)
p = make_poisson_posterior(n_structures,
                           data_file,
                           n_beads=n_beads, smooth_steepness=smooth_steepness,
                           beta=1.0,
                           disregard_lowest=0.0, lammda=1.0,
                           contact_distance=1.5,
                           k_ve=50.0, k2=100.0, k_bb=1000.0,
                           ignore_sequential_neighbors=2, include_zero_counts=True)
p = p.conditional_factory(weights=numpy.ones(n_structures),
                          # norm=ens_size/float(n_structures),
                          k2=100.0)
data = p.likelihoods['contacts'].forward_model.data_points
Es = -numpy.array([[p.likelihoods['contacts'].log_prob(**x.variables)
                    for x in y[-200::5]] for y in samples])
exec('Es_{}=Es'.format(n_structures))        

if not True:
    fig = plt.figure()
    for i in range(1, 6) + [20]:
        plt.plot(schedule, -eval('Es_{}'.format(i)).max(1),
                 label='n={}'.format(i), linewidth=2)
    plt.xlabel('beta')
    plt.ylabel('-log L(beta)')
    # plt.legend()
    plt.show()

if not True:
    from csbplus.statmech.wham import WHAM
    from csbplus.statmech.dos import DOS

    logZs = []
    
    ns_structures = [1, 2, 3, 5, 7, 10, 20]
    for x in [eval('Es_{}'.format(n_structures)) for n_structures in ns_structures]:
        # x = x[:,-200::5]
        q = numpy.outer(schedule, x.flatten())

        wham = WHAM(q.shape[1], q.shape[0])
        wham.N[:] = x.shape[1]
        wham.run(q, niter=10000, tol=1e-10, verbose=0)

        dos = DOS(x.flatten(), wham.s)
        print dos.log_Z(1) - dos.log_Z(0)
        logZs.append(dos.log_Z(1) - dos.log_Z(0))

    fig = plt.figure()
    plt.plot(ns_structures, logZs, '-o', linewidth=2)
    plt.xlabel('# of structures')
    plt.ylabel('log(evidence)')
    plt.show()
