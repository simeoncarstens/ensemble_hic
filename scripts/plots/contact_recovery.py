import os
import numpy

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import distance_matrix
from csb.bio.io.wwpdb import StructureParser

from ensemble_hic.analysis_functions import load_sr_samples
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.forward_models import EnsembleContactsFWM

data_dir = os.path.expanduser('~/projects/ensemble_hic/data/')
what = 'hairpin_s'
what = 'proteins'
burnin = 33000
cdistance = 8.5
show_plots = not False
write_ensembles = False
write_clusters = not True

if what == 'hairpin_s':
    sd = ((1,  28),
          (2,  30),
          (3,  36),
          (4,  40),
          (5,  45),
          (10, 58))
    n_structures, n_replicas = sd[1]

    base = '/scratch/scarste/ensemble_hic/hairpin_s/hairpin_s_fwm_poisson_new_'
    config_path = base + 'it2_{}structures_sn_{}replicas/'.format(n_structures,
                                                                  n_replicas)
    
    config_path += 'config.cfg'
    settings = parse_config_file(config_path)

    parser = StructureParser(data_dir + 'hairpin_s/hairpin.pdb') 
    true1 = parser.parse().get_coordinates(['CA'])
    parser = StructureParser(data_dir + 'hairpin_s/snake.pdb') 
    true2 = parser.parse().get_coordinates(['CA'])

    label1 = 'hairpin'
    label2 = 'snake'

if what == 'proteins':
    sd = ((1, 58),
          (2, 54),
          (3, 66),
          (4, 70),
          (5, 74),
          (10, 98))
    n_structures, n_replicas = sd[1]

    base = '/scratch/scarste/ensemble_hic/proteins/1pga_1shf_fwm_poisson_new_'
    config_path = base + 'it3_{}structures_sn_{}replicas/'.format(n_structures,
                                                                  n_replicas)

    # base = '/scratch/scarste/ensemble_hic/proteins/1pga_none_fwm_poisson_new_'
    # config_path = base + '1structures_sn_39replicas/'.format(n_structures,
    #                                                               n_replicas)
    # base = '/scratch/scarste/ensemble_hic/proteins/1shf_none_fwm_poisson_new_'
    # config_path = base + '1structures_sn_39replicas/'.format(n_structures,
    #                                                               n_replicas)
        
    config_path += 'config.cfg'
    settings = parse_config_file(config_path)

    parser = StructureParser(data_dir + what + '/1pga.pdb') 
    true1 = parser.parse().get_coordinates(['CA'])
    parser = StructureParser(data_dir + what + '/1shf.pdb') 
    true2 = parser.parse().get_coordinates(['CA'])

    label1 = '1PGA'
    label2 = '1SHF'
    label1 = 'GB1 domain'
    label2 = 'SH3 domain'

n_beads = len(true1)
samples_folder = settings['general']['output_folder'] + 'samples/'
n_replicas = int(settings['replica']['n_replicas'])
n_samples = int(settings['replica']['n_samples'])
dump_interval = int(settings['replica']['samples_dump_interval'])

samples = load_sr_samples(samples_folder, n_replicas, n_samples,
                          dump_interval, burnin)
structures = np.array([sample.variables['structures'].reshape(n_structures, -1, 3)
                       for sample in samples])
structures = structures.reshape(len(samples) * n_structures, -1, 3)
fake_data_points = np.array([[i,j,0] for i in range(n_beads)
                                     for j in range(i+1, n_beads)])
ana_fwm = EnsembleContactsFWM('petitprince', 1,
                              np.ones(len(fake_data_points)) * cdistance,
                              fake_data_points)

if True:
    true1_contacts = pdist(true1) < cdistance
    true2_contacts = pdist(true2) < cdistance
    normalization1 = float(n_beads * (n_beads - 1) / 2.0)
    normalization2 = float(n_beads * (n_beads - 1) / 2.0)

    cun1_mask = np.logical_and(true1_contacts, np.logical_not(true2_contacts))
    cun_1 = true1_contacts[cun1_mask]
    cun2_mask = np.logical_and(true2_contacts, np.logical_not(true1_contacts))
    cun_2 = true2_contacts[cun2_mask]
    ncun1_mask = np.logical_and(np.logical_not(true1_contacts), true2_contacts)
    ncun_1 = true1_contacts[ncun1_mask]
    ncun2_mask = np.logical_and(np.logical_not(true2_contacts), true1_contacts)
    ncun_2 = true2_contacts[ncun2_mask]
    
    cRMSDs_1 = 1-np.array([np.sum((cun_1 - (pdist(X) < cdistance)[cun1_mask]) ** 2)
                         for X in structures]) / float(sum(cun1_mask))
    cRMSDs_2 = 1-np.array([np.sum((cun_2 - (pdist(X) < cdistance)[cun2_mask]) ** 2)
                         for X in structures]) / float(sum(cun2_mask))
    ncRMSDs_1 = 1-np.array([np.sum((ncun_1 - (pdist(X) < cdistance)[ncun1_mask]) ** 2)
                         for X in structures]) / float(sum(ncun1_mask))
    ncRMSDs_2 = 1-np.array([np.sum((ncun_2 - (pdist(X) < cdistance)[ncun2_mask]) ** 2)
                         for X in structures]) / float(sum(ncun2_mask))

    cumulative = True
    if cumulative:
        weights = np.ones(len(structures)) / float(len(structures))
        
        fig = plt.figure()
        ax = fig.add_subplot(211)
        cRMSDs_1_sorted = np.sort(cRMSDs_1)
        ncRMSDs_1_sorted = np.sort(ncRMSDs_1)
        ax.plot(np.sort(cRMSDs_1), np.linspace(0, 1, len(cRMSDs_1), endpoint=False), 
                label='contacts')
        ax.plot(np.sort(ncRMSDs_1),np.linspace(0, 1, len(ncRMSDs_1), endpoint=False), 
                label='non-contacts')
        ax.set_ylabel('fraction of structures')
        #ax.set_xlabel('fraction of correct (non-)contacts to {}'.format(label1))
        ax.set_xticklabels(())
        ax.legend()
        ax.text(0.5, 0.8, label1)
        
        ax = fig.add_subplot(212, sharey=ax)
        ax.plot(np.sort(cRMSDs_2),np.linspace(0, 1, len(cRMSDs_2), endpoint=False), 
                label='contacts')
        ax.plot(np.sort(ncRMSDs_2),np.linspace(0, 1, len(ncRMSDs_2), endpoint=False), 
                label='non-contacts')
        ax.set_ylabel('fraction of structures')
        ax.set_xlabel('fraction of correct (non-)contacts')# to {}'.format(label2))
        ax.text(0.5, 0.8, label2)
        #ax.set_xticklabels((0.0,0.2,0.4,0.6,0.8,1.0))
        #ax.legend()        
    else:
        n_bins = int(sqrt(len(structures)))
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.hist(cRMSDs_1, bins=n_bins, range=(0,1), alpha=0.6, label='contacts')
        ax.hist(ncRMSDs_1, bins=n_bins, range=(0,1), alpha=0.6, label='non-contacts')
        ax.set_xlabel('fraction of correct (non-)contacts to {}'.format(label1))
        ax.set_ylabel('# structures')
        ax.set_yticks([])
        ax.legend()

        ax = fig.add_subplot(212)
        ax.hist(cRMSDs_2, bins=n_bins, range=(0,1), alpha=0.6, label='contacts')
        ax.hist(ncRMSDs_2, bins=n_bins, range=(0,1), alpha=0.6, label='non-contacts')
        ax.set_xlabel('fraction of correct (non-)contacts to {}'.format(label2))
        ax.set_ylabel('# structures')
        ax.set_yticks([])
        ax.legend()

    fig.tight_layout()
    if show_plots:
        plt.show()

    cRMSD_threshold = 0.6
    ncRMSD_threshold = 0.2
    mask1 = np.logical_and(cRMSDs_1 > cRMSD_threshold,
                           ncRMSDs_1 > ncRMSD_threshold)
    mask2 = np.logical_and(cRMSDs_2 > cRMSD_threshold,
                           ncRMSDs_2 > ncRMSD_threshold)
    mask1_wo_mask2 = np.logical_and(mask1, np.logical_not(mask2))
    mask2_wo_mask1 = np.logical_and(mask2, np.logical_not(mask1))
    mask_ambiguous = np.logical_and(np.logical_not(mask1_wo_mask2),
                                    np.logical_not(mask2_wo_mask1))
    cluster1 = structures[mask1_wo_mask2]
    cluster2 = structures[mask2_wo_mask1]
    ambiguous = structures[mask_ambiguous]

    if write_clusters:
        from ensemble_hic.analysis_functions import write_ensemble, write_VMD_script
        from ensemble_hic.setup_functions import make_posterior
        
        step = 10
        
        p = make_posterior(settings)
        bead_radii = p.priors['nonbonded_prior'].bead_radii
        
        article_path = os.path.expanduser('~/projects/ehic-paper/')
        for c in ('cluster1', 'cluster2', 'ambiguous'):
            filename = '{}_{}conformers_{}'.format(what, n_structures, c)
            pdb_path = article_path + 'clusters/' + filename + '.pdb'
            rc_path = article_path + 'clusters/' + filename + '.rc'
            exec('cluster = '+c)
            selection = cluster.reshape(-1, n_structures, n_beads, 3)
            selection = cluster[::step].reshape(-1,n_beads,3)
            from csb.bio.structure import InvalidOperation
            try:
                write_ensemble(selection, pdb_path)
                write_VMD_script(pdb_path, bead_radii, rc_path)
            except InvalidOperation as err:
                print "One of the clusters is probably empty; no files written"
                print "Error: {}".format(err)

if not True:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist([s.variables['norm'] for s in samples], bins=int(np.sqrt(len(samples))))
    ax.set_xlabel('gamma')
    ax.set_yticks([])
    fig.tight_layout()

    if show_plots:
        plt.show()
    

if not True:

    from csb.bio.utils import radius_of_gyration

    rgs = map(radius_of_gyration, structures)
    true1_rg = radius_of_gyration(true1)
    true2_rg = radius_of_gyration(true2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.hist(rgs, bins=int(sqrt(len(rgs))))
    ax.plot((true1_rg, true1_rg), (0, max(h[0])), c='r', label=label1)
    ax.plot((true2_rg, true2_rg), (0, max(h[0])), c='b', label=label2)
    ax.set_xlabel('radius of gyration')
    ax.set_ylabel('# structures')
    ax.set_yticks([])
    ax.legend()
    
    if show_plots:
        plt.show()

    
if not True:

    from csb.bio.utils import rmsd

    rgs = map(radius_of_gyration, structures)

    rmsds1 = np.array(map(lambda x: rmsd(true1, x), structures))
    rmsds2 = np.array(map(lambda x: rmsd(true2, x), structures))
    
    n_bins = 80
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.hist(rmsds1, n_bins, label=label1, alpha=0.6)
    ax.set_xlabel('RMSD to {}'.format(label1))
    ax.set_ylabel('# structures')
    
    ax = fig.add_subplot(222)
    ax.hist(rmsds2, n_bins, label=label2, alpha=0.6)
    ax.set_xlabel('RMSD to {}'.format(label2))
    ax.set_ylabel('# structures')

    structures_flipped1 = structures.copy()
    structures_flipped2 = structures.copy()
    structures_flipped1[np.logical_and(7.5 < rmsds1, rmsds1 < 10.0)] *= -1
    structures_flipped2[np.logical_and(7.5 < rmsds2, rmsds2 < 11.0)] *= -1
    rmsds_flipped1 = np.array(map(lambda x: rmsd(true1, x), structures_flipped1))
    rmsds_flipped2 = np.array(map(lambda x: rmsd(true2, x), structures_flipped2))

    ax = fig.add_subplot(223)
    ax.hist(rmsds_flipped1, n_bins, label=label1, alpha=0.6)
    ax.set_xlabel('RMSD to {} (2nd peak flipped)'.format(label1))
    ax.set_ylabel('fraction of structures')
    
    ax = fig.add_subplot(224)
    ax.hist(rmsds_flipped2, n_bins, label=label2, alpha=0.6)
    ax.set_xlabel('RMSD to {} (2nd peak flipped)'.format(label2))
    ax.set_ylabel('fraction of structures')

    fig.tight_layout()
    if show_plots:
        plt.show()


    
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(np.sort(rmsds1), np.linspace(0, 1, len(rmsds1), endpoint=False),
            label=label1)
    ax.plot(np.sort(rmsds2), np.linspace(0, 1, len(rmsds2), endpoint=False),
            label=label2)
    #ax.set_xlabel('RMSD')
    ax.set_ylabel('fraction of structures')
    ax.set_yticks((0.0,0.25,0.5,0.75,1))
    ax.set_xticks(())
    ax.legend()
    
    ax = fig.add_subplot(212)
    ax.plot(np.sort(rmsds_flipped1), np.linspace(0, 1, len(rmsds_flipped1),
                                                 endpoint=False),
            label=label1)
    ax.plot(np.sort(rmsds_flipped2), np.linspace(0, 1, len(rmsds_flipped2),
                                                 endpoint=False),
            label=label2)
    ax.set_xlabel('RMSD')
    #ax.set_ylabel('fraction of structures')
    ax.set_yticks((0.0,0.25,0.5,0.75,1))
    ax.legend()

    fig.tight_layout()
    if show_plots:
        plt.show()

if write_ensembles:
    from ensemble_hic.analysis_functions import write_ensemble, write_VMD_script
    from ensemble_hic.setup_functions import make_posterior

    step = 20

    p = make_posterior(settings)
    bead_radii = p.priors['nonbonded_prior'].bead_radii

    article_path = os.path.expanduser('~/projects/ehic-paper/')
    pdb_path = article_path + 'ensembles/{}_{}conformers.pdb'.format(what,
                                                                     n_structures)
    rc_path = article_path + 'ensembles/{}_{}conformers.rc'.format(what,
                                                                   n_structures)
    structures_selection = structures.reshape(-1, n_structures, n_beads, 3)
    structures_selection = structures_selection[::step].reshape(-1,n_beads,3)
    write_ensemble(structures_selection, pdb_path)
    write_VMD_script(pdb_path, bead_radii, rc_path)
