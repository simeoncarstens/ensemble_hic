import os
import sys
import numpy as np

from isd2.pdf.posteriors import Posterior
    
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.setup_functions import setup_weights
from ensemble_hic.analysis_functions import load_sr_samples

config_file = '/scratch/scarste/ensemble_hic/protein_1pga_1shf_mpdata_es100_sigma0.05/config.cfg'
config_file = '/scratch/scarste/ensemble_hic/bau5C_test/config.cfg'
settings = parse_config_file(config_file)
n_replicas = 40
target_replica = n_replicas


output_folder = settings['general']['output_folder']
if output_folder[-1] != '/':
    output_folder += '/'
n_beads = int(settings['general']['n_beads'])
n_structures = int(settings['general']['n_structures'])
schedule = np.load(output_folder + 'schedule.pickle')

settings['initial_state']['weights'] = setup_weights(settings)
posterior = make_posterior(settings)
posterior['lammda'].set(schedule['lammda'][target_replica -1])
posterior['beta'].set(schedule['beta'][target_replica -1])
p = posterior
variables = p.variables
L = posterior.likelihoods['ensemble_contacts']
data = L.forward_model.data_points

samples = load_sr_samples(output_folder + 'samples/', n_replicas, 20001, 100,
                          burnin=0000)
samples = samples[None,:]
if 'weights' in samples[-1,-1].variables:
    weights = np.array([x.variables['weights'] for x in samples.ravel()])
if 'norm' in samples[-1,-1].variables:
    norms = np.array([x.variables['norm'] for x in samples.ravel()])


if True:
    ## plot sampling statistics
    fig = plt.figure()

    ax = fig.add_subplot(321)
    energies = [np.load(output_folder + 'energies/replica{}.npy'.format(i+1))
                for i in range(len(schedule))]
    energies = np.array(energies).T
    plt.plot(energies.sum(1))
    plt.xlabel('MC samples?')
    plt.ylabel('extended ensemble energy')
    
    ax = fig.add_subplot(322)
    acceptance_rates = np.loadtxt(output_folder + 'statistics/re_stats.txt',
                                  dtype=float)
    acceptance_rates = acceptance_rates[-1,1:]
    x_interval = 5
    xticks = np.arange(0, n_replicas)[::-1][::x_interval][::-1]
    plt.plot(np.arange(0.5, n_replicas - 0.5), acceptance_rates, '-o')
    plt.xlabel('lambda')
    ax.set_xticks(xticks)
    lambda_labels = schedule['lammda'][::-1][::x_interval][::-1]
    lambda_labels = ['{:.2f}'.format(l) for l in lambda_labels]
    ax.set_xticklabels(lambda_labels)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xticks)
    plt.xlabel('beta')
    beta_labels = schedule['beta'][::-1][::x_interval][::-1]
    beta_labels = ['{:.2f}'.format(l) for l in beta_labels]
    ax2.set_xticklabels(beta_labels)
    plt.ylabel('acceptance rate')
    
    ax = fig.add_subplot(323)
    E_likelihood = -np.array([L.log_prob(**x.variables)
                              for x in samples[-1,:]])
    E_backbone = -np.array([p.priors['backbone_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])
    E_exvol = -np.array([p.priors['nonbonded_prior'].log_prob(structures=x.variables['structures']) for x in samples[-1,50:]])
    plt.plot(E_likelihood, linewidth=2, label='likelihood', color='blue')
    plt.plot(E_backbone, linewidth=2, label='backbone', color='green')
    plt.plot(E_exvol, linewidth=2, label='volume exclusion', color='red')
    leg = plt.legend()
    leg.draggable()
    plt.xlabel('replica transitions')
    plt.ylabel('target ensemble energies')
    
    ax = fig.add_subplot(324)
    hmc_acceptance_rates = np.loadtxt(output_folder + 'statistics/mcmc_stats.txt',
                          dtype=float)[:,2 * n_replicas]
    plt.plot(hmc_acceptance_rates)
    plt.xlabel('~# of MC samples')
    plt.ylabel('target ensemble HMC timestep')

    fig.tight_layout()
    plt.show()


if True:
    ## plot data IF and backcalculated IF matrix
    fig = plt.figure()
    def remove_zero_beads(m):
        nm = filter(lambda x: not np.all(np.isnan(x)), m)
        nm = np.array(nm).T
        nm = filter(lambda x: not np.all(np.isnan(x)), nm)
        nm = np.array(nm).T

        return nm

    ax = fig.add_subplot(121, axisbg='grey')
    from csb.bio.utils import distance_matrix
    vardict = {k: samples[-1,-1].variables[k] for k in samples[-1,-1].variables.keys()
               if k in L.forward_model.variables}
    rec = L.forward_model(**vardict)
    m = np.zeros((n_beads, n_beads)) * np.nan
    m[data[:,0], data[:,1]] = rec
    m[data[:,1], data[:,0]] = rec
    m = remove_zero_beads(m)
    ms = ax.matshow(np.log(m+1), cmap=plt.cm.jet)
    ax.set_title('reconstructed')
    cb = fig.colorbar(ms, ax=ax)
    cb.set_label('contact frequency')

    ax = fig.add_subplot(122, axisbg='grey')
    m = np.zeros((n_beads, n_beads)) * np.nan
    m[data[:,0],data[:,1]] = data[:,2]
    m[data[:,1],data[:,0]] = data[:,2]
    m = remove_zero_beads(m)
    ms = ax.matshow(np.log(m+1), cmap=plt.cm.jet)
    ax.set_title('data')
    cb = fig.colorbar(ms, ax=ax)
    cb.set_label('contact frequency')

    fig.tight_layout()
    plt.show()

if True:
    ## plot single structure (smoothened) contact matrices
    from csb.bio.utils import distance_matrix
    cds = p['contact_distance'].value
    a = p['smooth_steepness'].value
    s = lambda d, cd: 0.5 * a * (cd - d) / np.sqrt(1.0 + a * a * (cd - d) * (cd - d)) + 0.5
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
        md = [s(dm[data[j,0], data[j,1]], cds[j]) for j in range(len(data))]
        m = np.zeros((n_beads, n_beads))
        m[data[:,0], data[:,1]] = md
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
        
    fig.tight_layout()
    plt.show()


if True:
    ## write out a PDB file with structures and a VMD script file to display them
    ## properly. Run with "vmd -e whatever.rc"
    from csb.bio.utils import fit
    from ensemble_hic.analysis_functions import write_ensemble
    
    ens = np.array([x.variables['structures'].reshape(n_structures, n_beads, 3)
                       for x in samples[-1,-1000::10]]).reshape(-1,n_beads,3)
    ens -= ens.mean(1)[:,None,:]

    if True:
        ## align to last sample, last structure
        cs = ens[-1]
        
        Rts = [fit(cs, x) for x in ens]
        aligned_ens = [np.dot(x, Rts[i][0].T) + Rts[i][1] for i, x in enumerate(ens)]
        ens = np.array(aligned_ens)
        
    write_ensemble(ens, '/tmp/out.pdb'.format(variables))

    ## make VMD startup script
    lines = ['color Display Background white',
             'menu main on',
             'menu graphics on',
             'mol load pdb /tmp/out.pdb',
             'mol color Index',
             'mol delrep 0 0',
             'mol representation VDW',
             'mol addrep 0'
            ]

    bead_radii = p.priors['nonbonded_prior'].bead_radii
    for i, r in enumerate(bead_radii):
        lines.append('set sel [atomselect top "index {}"]'.format(i))
        lines.append('$sel set radius {}'.format(r))

    with open('/tmp/bau5C.rc','w') as opf:
        [opf.write(line + '\n') for line in lines]
