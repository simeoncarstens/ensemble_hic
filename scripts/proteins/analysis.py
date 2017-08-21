import numpy, os
os.chdir(os.path.expanduser('~/projects/hic/py/hicisd2/ensemble_scripts/protein_test/'))
import matplotlib.pyplot as plt
from misc import kth_diag_indices, make_posterior, make_poisson_posterior, make_lognormal_posterior
from csb.bio.io import StructureParser

prot1 = '1pga'
prot2 = '1shf'

prot1 = '1ubq'
prot2 = '2ma1'

n_structures = 2
n_replicas = 40
ens_size = 200
variables = 'sn'
em = 'poisson'
isn = 1
smooth_steepness = 14.0
bla = '_cutoff5'
bla = '_wzeros_cutoff5'
bla = '_wzeros2'
bla = '_wzeros_cd1.8ss50_kbb1000_pdata_es100'
bla = '_wzeros_cd1.8ss14_kbb1000_mpdata_es100_sigma0.05'
# bla = '_wzeros_cd1.8ss14_kbb1000_mpdata_sigma0.05'
#bla = '_wzeros'
#bla = ''
include_zero_counts = True
sim_name = 'protein_isn{}{}_{}_{}_{}_{}'.format(isn, bla, prot1, prot2, em, variables)

sim_path = '/scratch/scarste/ensemble_hic/{}_{}structures_{}replicas/'.format(sim_name, n_structures, n_replicas)
# schedule = numpy.load(sim_path + 'schedule.pickle')['beta']
schedule = numpy.load(sim_path + 'schedule.pickle')['lammda']

data_file = '{}_{}_maxwell_poisson_data_es100_sigma0.05.txt'.format(prot1, prot2)
# data_file = '{}_{}_maxwell_poisson_data_sigma0.05.txt'.format(prot1, prot2)
# data_file = '{}_{}_poisson_data.txt'.format(prot1, prot2)
coords  = StructureParser(prot1 + '.pdb').parse().get_coordinates(['CA']) / 4.0
coords2 = StructureParser(prot2 + '.pdb').parse().get_coordinates(['CA']) / 4.0
n_beads = len(coords)

U = numpy.array([numpy.load(sim_path + 'energies/replica{}.npy'.format(i+1)) for i in range(len(schedule))]).T
pacc = numpy.loadtxt(sim_path + 'statistics/re_stats.txt', dtype=float)[-1,1:]
if True:
    from hicisd2.hicisd2lib import load_samples, load_sr_samples

    if True:
        samples = load_sr_samples(sim_path + 'samples/', n_replicas, 15001, 100,0)
        samples = samples[None,:]
    if 'k2' in samples[-1,-1].variables:
        k2s = numpy.array([x.variables['k2'] for x in samples.ravel()])
    if 'weights' in samples[-1,-1].variables:
        weights = numpy.array([x.variables['weights'] for x in samples.ravel()])
    if 'norm' in samples[-1,-1].variables:
        norms = numpy.array([x.variables['norm'] for x in samples.ravel()])

if em == 'poisson':
    p = make_poisson_posterior(n_structures,
                               data_file,
                               n_beads=n_beads, smooth_steepness=14.0, beta=1.0,
                               disregard_lowest=0.0, lammda=1.0,
                               contact_distance=2.0,
                               k_ve=50.0, k2=100.0,
                               ignore_sequential_neighbors=isn,
                               include_zero_counts=include_zero_counts,
                               k_bb=1000.0)
    

if variables == 'sw':
    p = p.conditional_factory(norm=1.0, k2=100.0)
elif variables == 'sn':
    p = p.conditional_factory(weights=numpy.ones(n_structures), k2=100.0)
elif variables == 's':
    p = p.conditional_factory(weights=numpy.ones(n_structures),
                              norm=ens_size/float(n_structures),
                              k2=100.0)
elif variables == 'sk2':
    p = p.conditional_factory(weights=numpy.ones(n_structures),
                              norm=ens_size/float(n_structures))
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
    X1 = load_pdb('1shf.pdb')
    X2 = load_pdb('1pga.pdb')
    X1 = coords
    X2 = coords2
    sigma = 0.1
    X = numpy.array([X1 + numpy.random.normal(size=X1.shape, scale=sigma) for _ in range(n_structures / 2)])
    X = numpy.vstack((X, [X2 + numpy.random.normal(size=X2.shape, scale=sigma) for _ in range(n_structures / 2)]))
    vardict = dict(structures=X)
    if variables == 'sw':
        oweights = numpy.ones(n_structures) * ens_size / float(n_structures)
        vardict.update(weights=oweights)
    elif variables == 'sn':
        norm = ens_size / float(n_structures)
        vardict.update(norm=norm)
    optimal_state = ISDState(vardict)

    plt.plot(-numpy.ones(len(E_likelihood)) * p.likelihoods['contacts'].log_prob(**optimal_state.variables), label='likelihood (optimal state)', color='blue', linewidth=2)
    plt.plot(-numpy.ones(len(E_likelihood)) * p.priors['backbone_prior'].log_prob(structures=optimal_state.variables['structures']), label='backbone (optimal state)', color='green', linewidth=2)
    plt.plot(-numpy.ones(len(E_likelihood)) * p.priors['boltzmann_prior'].log_prob(structures=optimal_state.variables['structures']), label='volume exclusion (optimal state)', color='red', linewidth=2)        

    plt.xlabel('replica transitions')
    plt.ylabel('energy')
    #plt.legend()
    plt.show()

if True:
    if 'k2' in samples[-1,-1].variables:
        ax = fig.add_subplot(326)
        ax.hist(k2s[500:], bins=30)
        plt.xlabel('precision')

plt.tight_layout()

if False:

    highp_contacts = dms[:,data[-23:,1], data[-23:,2]]

if True:
    from protlib import writeGNMtraj

    ens = numpy.array([x.variables['structures'].reshape(n_structures, n_beads, 3)
                       for x in samples[-1,-100::10]]).reshape(-1,n_beads,3)
    ens -= ens.mean(1)[:,None,:]

    # from csb.bio.utils import bfit
    # if True:
    #     ## align to prot1
    #     cs = coords
    # else:
    #     ## align to prot2
    #     cs = coords2
    # cs -= cs.mean(0)[None,:]
    
    
    # Rts = [bfit(cs, x) for x in ens]
    # aligned_ens = [numpy.dot(x, Rts[i][0].T) + Rts[i][1] for i, x in enumerate(ens)]
    # aligned_ens = numpy.array(aligned_ens)
    aligned_ens = ens
        
    writeGNMtraj(aligned_ens * 4.0, '/tmp/out_{}{}.pdb'.format(variables, bla))


if False:
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(data[:,0], label='data')
    fwm = p.likelihoods['contacts'].forward_model
    em = p.likelihoods['contacts'].forward_model
    md = fwm(**{k: v for k, v in samples[-1,-1].variables.iteritems()
                if k in fwm.variables})
    ax.plot(md, label='mock data')
    ax.legend()

    ax = fig.add_subplot(212)
    chi2 = lambda md, d: numpy.log(d/md)**2
    chi2s = numpy.array([chi2(md[i], data[i,0]) for i in range(len(md))])
    ax.plot(chi2s)

    plt.show()


if True:
    from csb.bio.utils import distance_matrix
    cd = p['contact_distance'].value
    a = p['smooth_steepness'].value
    s = lambda d: 0.5 * a * (cd - d) / numpy.sqrt(1.0 + a * a * (cd - d) * (cd - d)) + 0.5
    fig = plt.figure()
    for i in range(n_structures):
        ax = fig.add_subplot(4,5,i+1)
        dm = distance_matrix(samples[-1,-1].variables['structures'].reshape(n_structures, n_beads, 3)[i])
        md = s(dm[data[:,1], data[:,2]])
        m = numpy.zeros((n_beads, n_beads))
        m[data[:,1], data[:,2]] = md
        ms = ax.matshow(m+m.T)
        fig.colorbar(ms, ax=ax)
        if 'weights' in samples[-1,-1].variables:
            weights = samples[-1,-1].variables['weights']
        else:
            weights = p['weights'].value
        ax.set_title('w={:.1e}'.format(weights[i]))
    plt.show()
        
