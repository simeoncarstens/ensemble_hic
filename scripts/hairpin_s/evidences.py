import numpy
import matplotlib.pyplot as plt
from misc import kth_diag_indices, make_posterior, make_poisson_posterior, make_lognormal_posterior


n_structures = 100
n_replicas = 40
n_beads = 23
ens_size = 10000
variables = 's'
em = 'poisson'
sim_name = 'lambdatempering2_sphere_{}_{}'.format(em, variables)

sim_path = '/scratch/scarste/ensemble_hic/{}_{}structures_{}replicas/'.format(sim_name, n_structures, n_replicas)
schedule = numpy.load(sim_path + 'schedule.pickle')['beta']

data_file = 'spiral_hairpin_data_{}beads_littlenoise.txt'.format(n_beads)

U = numpy.array([numpy.load(sim_path + 'energies/replica{}.npy'.format(i+1)) for i in range(len(schedule))]).T

from hicisd2.hicisd2lib import load_samples, load_sr_samples
samples = load_samples(sim_path + 'samples/', n_replicas, 13501, 100, burnin=3000)
p = make_poisson_posterior(n_structures,
                           data_file,
                           n_beads=n_beads, smooth_steepness=5.0, beta=1.0,
                           disregard_lowest=0.0, lammda=1.0,
                           contact_distance=2.0,
                           k_ve=50.0, k2=100.0)
p = p.conditional_factory(weights=numpy.ones(n_structures),
                          norm=ens_size/float(n_structures),
                          k2=100.0)
data = p.likelihoods['contacts'].forward_model.data_points
Es = -numpy.array([[p.likelihoods['contacts'].log_prob(**x.variables)
                    for x in y] for y in samples])
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
    
    ns_structures = [1, 2, 3, 4, 5, 8, 12, 16, 20, 50, 100]
    for x in [eval('Es_{}'.format(n_structures)) for n_structures in ns_structures]:
        x = x[:,-100::10]
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
