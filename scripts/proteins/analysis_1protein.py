import numpy
import matplotlib.pyplot as plt
from misc import kth_diag_indices, make_posterior, make_poisson_posterior, make_lognormal_posterior
from csb.bio.io import StructureParser
from csb.bio.utils import fit
from hicisd2.hicisd2lib import load_samples, load_sr_samples

prot = '1shf'

n_structures = 1
n_replicas = 40
ens_size = 100
smooth_steepness = 14.0
if ens_size == 100:
    bla = '_wzeros_cd1.8ss50_kbb1000_pdata_es100'
    bla = '_wzeros_cd1.8ss14_kbb1000_mpdata_es100_sigma0.05'
    data_file = '{}_none_maxwell_poisson_data_es100.txt'.format(prot)
else:
    bla = '_wzeros_cd1.8ss50_kbb1000_pdata'
    data_file = '{}_none_poisson_data.txt'.format(prot)
sim_name = 'protein_isn1{}_{}_none_poisson_s'.format(bla, prot)
sim_path = '/scratch/scarste/ensemble_hic/{}_{}structures_{}replicas/'.format(sim_name, n_structures, n_replicas)
schedule = numpy.load(sim_path + 'schedule.pickle')['lammda']

coords  = StructureParser(prot + '.pdb').parse().get_coordinates(['CA']) / 4.0
n_beads = len(coords)

U = numpy.array([numpy.load(sim_path + 'energies/replica{}.npy'.format(i+1)) for i in range(len(schedule))]).T
pacc = numpy.loadtxt(sim_path + 'statistics/re_stats.txt', dtype=float)[-1,1:]

samples = load_sr_samples(sim_path + 'samples/', n_replicas, 20001, 100,0)
samples = samples[None,:]

p = make_poisson_posterior(n_structures,
                           data_file,
                           n_beads=n_beads, smooth_steepness=smooth_steepness,
                           beta=1.0,
                           disregard_lowest=0.0, lammda=1.0,
                           contact_distance=2.0,
                           k_ve=50.0, k2=100.0,
                           ignore_sequential_neighbors=1,
                           include_zero_counts=True,
                           k_bb=1000.0)
p = p.conditional_factory(weights=numpy.ones(n_structures),
                          norm=ens_size/float(n_structures),
                          k2=100.0)
data = p.likelihoods['contacts'].forward_model.data_points


fig = plt.figure()
if True:
    
    ax = fig.add_subplot(321)
    plt.plot(U.sum(1))
    plt.xlabel('MC samples?')
    plt.ylabel('extended ensemble energy')
        
    ax = fig.add_subplot(322)
    plt.plot(0.5 * (schedule[1:] + schedule[:-1]), pacc, '-o')
    plt.xlabel('replica parameter')
    plt.ylabel('acceptance rate')
    
    plt.show()

if True:

    ax = fig.add_subplot(324)
    from csb.bio.utils import distance_matrix
    vardict = {k: samples[-1,-1].variables[k] for k in samples[-1,-1].variables.keys()
               if k in p.likelihoods['contacts'].forward_model.variables}
    rec = p.likelihoods['contacts'].forward_model(**vardict)
    m = numpy.zeros((n_beads, n_beads))
    m[data[:,1], data[:,2]] = rec
    m[data[:,2], data[:,1]] = rec
    ms = ax.matshow(m)
    ax.set_title('reconstructed')
    cb = fig.colorbar(ms, ax=ax)
    cb.set_label('contact frequency')

    ax = fig.add_subplot(323)
    m = numpy.zeros((n_beads, n_beads))
    m[data[:,1],data[:,2]] = data[:,0]
    m[data[:,2],data[:,1]] = data[:,0]
    ms = ax.matshow(m)
    ax.set_title('data')
    cb = fig.colorbar(ms, ax=ax)
    cb.set_label('contact frequency')

if True:
    
    E_likelihood = -numpy.array([p.likelihoods['contacts'].log_prob(**x.variables)
                                 for x in samples[-1,:]])
    E_backbone = -numpy.array([p.priors['backbone_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,:]])
    E_exvol = -numpy.array([p.priors['boltzmann_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,:]])

    ax = fig.add_subplot(325)
    plt.plot(E_likelihood, linewidth=2, label='likelihood', color='blue')
    plt.plot(E_backbone, linewidth=2, label='backbone', color='green')
    plt.plot(E_exvol, linewidth=2, label='volume exclusion', color='red')

    ## create an optimal state
    from misc import load_pdb
    from isd2.samplers import ISDState
    X1 = coords
    sigma = 0.1
    X = X1 + numpy.random.normal(size=X1.shape, scale=sigma)
    vardict = dict(structures=X)
    optimal_state = ISDState(vardict)

    plt.plot(-numpy.ones(len(E_likelihood)) * p.likelihoods['contacts'].log_prob(**optimal_state.variables), label='likelihood (optimal state)', color='blue', linewidth=2)
    plt.plot(-numpy.ones(len(E_likelihood)) * p.priors['backbone_prior'].log_prob(structures=optimal_state.variables['structures']), label='backbone (optimal state)', color='green', linewidth=2)
    plt.plot(-numpy.ones(len(E_likelihood)) * p.priors['boltzmann_prior'].log_prob(structures=optimal_state.variables['structures']), label='volume exclusion (optimal state)', color='red', linewidth=2)        

    plt.xlabel('replica transitions')
    plt.ylabel('energy')
    #plt.legend()
    plt.show()

if True:
    from csb.bio.utils import rmsd
    from scipy.cluster.vq import kmeans2
    
    ens = numpy.array([x.variables['structures'].reshape(n_structures, n_beads, 3)
                       for x in samples[-1,100::5]]).reshape(-1,n_beads,3)
    cs = coords
    Rts = [fit(cs, x) for x in ens]
    aligned_ens = [numpy.dot(x, Rts[i][0].T) + Rts[i][1] for i, x in enumerate(ens)]
    aligned_ens = numpy.array(aligned_ens)
    rmsds = numpy.array(map(lambda x: rmsd(x, cs), aligned_ens))
    clusters = kmeans2(rmsds, 2)
    ax = fig.add_subplot(326)
    ax.hist(rmsds)
    plt.xlabel('RMSD')

plt.tight_layout()

if True:
    from protlib import writeGNMtraj

    ens = numpy.array([x.variables['structures'].reshape(n_structures, n_beads, 3)
                       for x in samples[-1,-100::10]]).reshape(-1,n_beads,3)
    ens -= ens.mean(1)[:,None,:]

    cs = coords
       
    Rts = [bfit(cs, x) for x in ens]
    aligned_ens = [numpy.dot(x, Rts[i][0].T) + Rts[i][1] for i, x in enumerate(ens)]
    aligned_ens = numpy.array(aligned_ens)
        
    writeGNMtraj(aligned_ens * 4.0, '/tmp/out_s{}.pdb'.format(bla))


