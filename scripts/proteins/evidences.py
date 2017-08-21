import numpy, os
os.chdir(os.path.expanduser('~/projects/hic/py/hicisd2/ensemble_scripts/protein_test/'))
import matplotlib.pyplot as plt
from misc import kth_diag_indices, make_posterior, make_poisson_posterior, make_lognormal_posterior
from csb.bio.io import StructureParser


n_structures = 1
n_replicas = 40
ens_size = 200
variables = 's'
em = 'poisson'
smooth_steepness = 14.0

prot1 = '1pga'
prot2 = '1shf'

prot1 = '1ubq'
prot2 = '2ma1'

isn = 1
bla = '_wzeros_cd1.8ss14_kbb1000_mpdata_es100_sigma0.05'
sim_name = 'protein_isn{}{}_{}_{}_{}_{}'.format(isn, bla, prot1, prot2, em, variables)

sim_path = '/scratch/scarste/ensemble_hic/{}_{}structures_{}replicas/'.format(sim_name, n_structures, n_replicas)
schedule = numpy.load(sim_path + 'schedule.pickle')['lammda']

data_file = '{}_{}_maxwell_poisson_data_es100_sigma0.05.txt'.format(prot1, prot2)

coords  = StructureParser(prot1 + '.pdb').parse().get_coordinates(['CA'])
n_beads = len(coords)

from hicisd2.hicisd2lib import load_samples, load_sr_samples
samples = load_samples(sim_path + 'samples/', n_replicas, 20001, 100, burnin=5000)
p = make_poisson_posterior(n_structures,
                           data_file,
                           n_beads=n_beads, smooth_steepness=smooth_steepness,
                           beta=1.0,
                           disregard_lowest=0.0, lammda=1.0,
                           contact_distance=1.8,
                           k_ve=50.0, k2=100.0, k_bb=1000.0,
                           ignore_sequential_neighbors=isn, include_zero_counts=True)
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
    
    ns_structures = [1, 2, 3, 4, 5, 6, 8]#, 10, 20]
    for x in [eval('Es_{}'.format(n_structures)) for n_structures in ns_structures]:
        x = x[:,-400::10]
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
