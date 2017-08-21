import numpy, os
os.chdir(os.path.expanduser('~/projects/hic/py/hicisd2/ensemble_scripts/bau5C_test/'))
import matplotlib.pyplot as plt
from misc import kth_diag_indices, make_poisson_posterior

dataset = 'K562'
dataset = 'GM12878'

n_structures = 5
n_replicas = 80
ens_size = 200 * 67.5
variables = 'sn'
em = 'poisson'
smooth_steepness = 10.0
sim_name = 'bau5C_isn2_3rdrootradii_smallersphere_betatempering_{}_{}_{}'.format(dataset, em, variables)

sim_path = '/scratch/scarste/ensemble_hic/{}_{}structures_{}replicas/'.format(sim_name, n_structures, n_replicas)
schedule = numpy.load(sim_path + 'schedule.pickle')['lammda']

data_file = os.path.expanduser('~/projects/hic/data/bau2011/{}.txt'.format(dataset))
n_beads = 70

U = numpy.array([numpy.load(sim_path + 'energies/replica{}.npy'.format(i+1)) for i in range(len(schedule))]).T
pacc = numpy.loadtxt(sim_path + 'statistics/re_stats.txt', dtype=float)[-1,1:]
if True:
    from hicisd2.hicisd2lib import load_samples, load_sr_samples

    if True:
        samples = load_sr_samples(sim_path + 'samples/',
                                  n_replicas,
                                  2101, 100,
                                  burnin=0000)
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
                               n_beads=n_beads,
                               smooth_steepness=smooth_steepness, beta=1.0,
                               disregard_lowest=0.0, lammda=1.0,
                               contact_distance=1.5,
                               k_ve=50.0, k2=100.0,
                               ignore_sequential_neighbors=2,
                               include_zero_counts=True,
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
    # plt.plot(0.5 * (schedule[1:] + schedule[:-1]), pacc, '-o')
    plt.plot(pacc, '-o')
    # plt.xlabel('replica parameter')
    plt.xlabel('replica index')
    plt.ylabel('acceptance rate')
    
    plt.show()

if True:

    def remove_zero_beads(m):
        nm = filter(lambda x: not numpy.all(numpy.isnan(x)), m)
        nm = numpy.array(nm).T
        nm = filter(lambda x: not numpy.all(numpy.isnan(x)), nm)
        nm = numpy.array(nm).T

        return nm

    ax = fig.add_subplot(324, axisbg='grey')
    from csb.bio.utils import distance_matrix
    vardict = {k: samples[-1,-1].variables[k] for k in samples[-1,-1].variables.keys()
               if k in p.likelihoods['contacts'].forward_model.variables}
    rec = p.likelihoods['contacts'].forward_model(**vardict)
    m = numpy.zeros((n_beads, n_beads)) * numpy.nan
    m[data[:,1], data[:,2]] = rec
    m[data[:,2], data[:,1]] = rec
    m = remove_zero_beads(m)
    ms = ax.matshow(numpy.log(m+1), cmap=plt.cm.jet)
    ax.set_title('reconstructed')
    cb = fig.colorbar(ms, ax=ax)
    cb.set_label('contact frequency')

    ax = fig.add_subplot(323, axisbg='grey')
    m = numpy.zeros((n_beads, n_beads)) * numpy.nan
    m[data[:,1],data[:,2]] = data[:,0]
    m[data[:,2],data[:,1]] = data[:,0]
    m = remove_zero_beads(m)
    ms = ax.matshow(numpy.log(m+1), cmap=plt.cm.jet)
    ax.set_title('data')
    cb = fig.colorbar(ms, ax=ax)
    cb.set_label('contact frequency')

if True:
    
    E_likelihood = -numpy.array([p.likelihoods['contacts'].log_prob(**x.variables)
                                 for x in samples[-1,50:]])
    E_backbone = -numpy.array([p.priors['backbone_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])
    E_exvol = -numpy.array([p.priors['boltzmann_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])

    ax = fig.add_subplot(325)
    plt.plot(E_likelihood, linewidth=2, label='likelihood', color='blue')
    plt.plot(E_backbone, linewidth=2, label='backbone', color='green')
    plt.plot(E_exvol, linewidth=2, label='volume exclusion', color='red')

    plt.xlabel('replica transitions')
    plt.ylabel('energy')
    #plt.legend()

if True:

    hmc_pacc = numpy.loadtxt(sim_path + 'statistics/mcmc_stats.txt',
                             dtype=float)[:,2 * n_replicas]

    ax = fig.add_subplot(326)
    plt.plot(hmc_pacc)
    plt.xlabel('~# of MC samples')
    plt.ylabel('target ensemble HMC timestep')

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
    from csb.bio.utils import fit
    
    ens = numpy.array([x.variables['structures'].reshape(n_structures, n_beads, 3)
                       for x in samples[-1,-1000::10]]).reshape(-1,n_beads,3)
    ens -= ens.mean(1)[:,None,:]

    if True:
        ## align to prot1
        cs = ens[-1]
        
    Rts = [fit(cs, x) for x in ens]
    aligned_ens = [numpy.dot(x, Rts[i][0].T) + Rts[i][1] for i, x in enumerate(ens)]
    aligned_ens = numpy.array(aligned_ens)
        
    writeGNMtraj(aligned_ens, '/tmp/out_bau5C_{}.pdb'.format(variables))

    ## make VMD startup script
    lines = ['color Display Background white',
             'menu main on',
             'menu graphics on',
             'mol load pdb out_bau5C_{}.pdb'.format(variables),
             'mol color Index',
             'mol delrep 0 0',
             'mol representation VDW',
             'mol addrep 0'
            ]


    bead_radii = p.priors['boltzmann_prior'].forcefield.radii
    for i, r in enumerate(bead_radii):
        lines.append('set sel [atomselect top "index {}"]'.format(i))
        lines.append('$sel set radius {}'.format(r))

    with open('/tmp/bau5C.rc','w') as opf:
        [opf.write(line + '\n') for line in lines]
        
if True:
    from csb.bio.utils import distance_matrix
    cds = p['contact_distance'].value
    a = p['smooth_steepness'].value
    s = lambda d, cd: 0.5 * a * (cd - d) / numpy.sqrt(1.0 + a * a * (cd - d) * (cd - d)) + 0.5
    for i in range(n_structures):
        if i == 0:
            fig = plt.figure()
            ax = fig.add_subplot(5, 5, 1)
        elif i % 25 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(5, 5, 1)
        else:
            ax = fig.add_subplot(5, 5, i % 25 + 1)
        dm = distance_matrix(samples[-1,-1].variables['structures'].reshape(n_structures, n_beads, 3)[i])
        md = [s(dm[data[j,1], data[j,2]], cds[j]) for j in range(len(data))]
        m = numpy.zeros((n_beads, n_beads))
        m[data[:,1], data[:,2]] = md
        ms = ax.matshow(m+m.T, cmap=plt.cm.jet)
        cb = fig.colorbar(ms, ax=ax)
        # if 'weights' in samples[-1,-1].variables:
        #     weights = samples[-1,-1].variables['weights']
        # else:
        #     weights = p['weights'].value
        # ax.set_title('w={:.1e}'.format(weights[i]))
        ax.set_title('{}'.format(i))
        ax.set_xticks([])
        ax.set_yticks([])
        # cb.set_ticks([])
    plt.show()

if not True:
    fig = plt.figure()

    ax = fig.add_subplot(211)
    bbds = sqrt(sum((ens[:,1:] - ens[:,:-1]) ** 2, 2))
    bbtds = bead_radii[1:]+bead_radii[:-1]
    ax.scatter(numpy.arange(0, n_beads-1), bbtds, c='r')
    ax.boxplot(bbds, positions=numpy.arange(0, n_beads-1))
    ax.set_xticks(numpy.arange(-0.5, n_beads - 1.5)[::2])
    ax.set_xticklabels([str(x) for x in numpy.arange(70)[::2]])
    ax.set_xlabel('bead index')
    ax.set_ylabel('distance')

    if False:
        ax = fig.add_subplot(212)
        from scipy.spatial.distance import pdist, squareform
        ds = numpy.array(map(pdist, ens))
        tds = numpy.add.outer(bead_radii, bead_radii)
        tds[numpy.diag_indices(len(tds))] = 0
        tds = squareform(tds)
        r = numpy.arange(1, len(tds)+1)
        ax.scatter(r, tds, c='r')
        ax.scatter(r, ds.mean(0))
        ax.scatter(r, ds.mean(0) + 0.5 * ds.std(0), c='g')
        ax.scatter(r, ds.mean(0) - 0.5 * ds.std(0), c='g')
        ax.set_xticks([])
        ax.set_ylabel('distance')
        ax.set_xlabel('bead pairs')
    
    plt.show()

if True:
    PR = 0  # contains promoter
    AG = 1  #          active gene
    NA = 2  #          non-active gene
    HS = 3  #          DNase1 hypersensitivity site
    CT = 4  #          CTCF site
    HM = 5  #          H3K4me3 site
    annotations = numpy.loadtxt(os.path.expanduser('~/projects/hic/data/bau2011/annotations.txt'),
                                skiprows=2, dtype=int)
    annotations = annotations[:, (0,1,2,3,4,5) if dataset == 'GM12878'
                              else (6,7,8,9,10,11)]
    PR_beads = numpy.where(annotations[:,PR] == 1)[0]
    AG_beads = numpy.where(annotations[:,AG] == 1)[0]
    NA_beads = numpy.where(annotations[:,NA] == 1)[0]
    HS_beads = numpy.where(annotations[:,HS] == 1)[0]
    CT_beads = numpy.where(annotations[:,CT] == 1)[0]
    HM_beads = numpy.where(annotations[:,HM] == 1)[0]

    ## one RF contains the HS40 site, an enhancer
    ## (for the alpha-globin genes, I guess)
    HS40_bead = 20

    ## one bead contains the alpha-globin gene cluster
    aglobin_bead = 26 ## according to Bau et al. (2011) SI
    # aglobin_bead = 28 ## according to Wikipedia

    ## Bau et al. (2011) measure smaller distances between HS40
    ## and alpha-globin beads in K562 cells, in which alpha-globin
    ## is transcribed. In GM12878, it is repressed and they measure
    ## longer distances

    ens = numpy.array([x.variables['structures'].reshape(n_structures, n_beads, 3)
                       for x in samples[-1,::10]]).reshape(-1,n_beads,3)
    ens -= ens.mean(1)[:,None,:]
    if True:
        ## use only structures below radius of gyration threshold
        ## This excludes stretched structures with little contacts
        from csb.bio.utils import radius_of_gyration
        rg_threshold = bead_radii.mean() * 8
        rgs = numpy.array(map(radius_of_gyration, ens))
        ens = ens[rgs < rg_threshold]
        
    HS40_aglobin_distances = numpy.sqrt(numpy.sum((ens[:,HS40_bead] - ens[:,aglobin_bead])**2, 1))

    ## now estimate density along the fiber
    density_cutoff = bead_radii.mean() * 3
    densities = []
    for i in range(n_beads):
        ds = numpy.sqrt(numpy.sum((ens[:,i][:,None] - ens) ** 2, 2))
        densities.append(sum(ds < density_cutoff) / float(len(ens) * n_beads))

    ## calculate and distances between alpha-globin bead and all others
    ag_all_ds = numpy.sqrt(numpy.sum((ens[:,aglobin_bead][:,None] - ens) ** 2, 2))

    if dataset == 'K562':
        ds_K562 = HS40_aglobin_distances
        ag_all_ds_K562 = ag_all_ds
        densities_K562 = densities
    else:
        ds_GM12878 = HS40_aglobin_distances
        ag_all_ds_GM12878 = ag_all_ds
        densities_GM12878 = densities


    if 'ds_K562' in dir() and 'ds_GM12878' in dir():

        fig = plt.figure()
        ax = fig.add_subplot(221)
        #ax.boxplot(numpy.vstack((ds_K562, ds_GM12878)).T)
        bpl = ax.boxplot([ds_K562, ds_GM12878], notch=True, patch_artist=True)
        ax.set_xticklabels(['K562', 'GM12878'])
        colors = ['red', 'blue']
        for patch, color in zip(bpl['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel('distance between HS40 and alpha-globin')
        
        ax = fig.add_subplot(222)
        ax.plot(range(1,n_beads+1), mean(ag_all_ds_K562, 0),
                c='r', label='mean (K562)')
        ax.plot(range(1,n_beads+1), mean(ag_all_ds_GM12878, 0),
                c='b', label='mean (GM12878)')
        bpl = ax.boxplot(ag_all_ds_K562, patch_artist=True)
        colors = ['red'] * len(bpl['boxes'])
        for patch, color in zip(bpl['boxes'], colors):
            patch.set_facecolor(color)

        bpl = ax.boxplot(ag_all_ds_GM12878, patch_artist=True)
        colors = ['blue'] * len(bpl['boxes'])
        for patch, color in zip(bpl['boxes'], colors):
            patch.set_facecolor(color)
        ax.plot([HS40_bead, HS40_bead], [0, ax.get_ylim()[1]], color='green', label='HS40 bead')
        ax.set_xlabel('bead index')
        ax.set_ylabel('distance to alpha-globin bead')
        ax.set_xticks(numpy.arange(1, n_beads+1, 5))
        ax.set_xticklabels([str(x-1) for x in numpy.arange(1, n_beads+1, 5)])
        ax.legend()

        ax = fig.add_subplot(223)
        ax.plot(densities_K562, label='K562', lw=2, c='r')
        ax.plot(densities_GM12878, label='GM12878', lw=2, c='b')
        ax.set_xlabel('bead index')
        ax.set_ylabel('fraction of beads within ({:.1f} * avg bead radius) units'.format(density_cutoff))
        ax.legend()
        plt.show()
