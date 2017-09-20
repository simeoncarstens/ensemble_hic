import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from csb.bio.utils import rmsd, radius_of_gyration

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_sr_samples

n_replicas = 106
burnin = 20000
save_figures = not True
    
if not True:
    config_file_GM12878 = sys.argv[1]
    config_file_K562 = sys.argv[2]
else:
    path = '/scratch/scarste/ensemble_hic/bau2011/'
    config_file_GM12878 = path + 'GM12878_10structures_s_106replicas_nosphere_optsched3/config.cfg'
    config_file_K562 = path + 'K562_10structures_s_106replicas_nosphere_optsched3/config.cfg'
    # config_file_GM12878 = path + 'GM12878_8structures_sn_40replicas/config.cfg'
    # config_file_K562 = path + 'K562_8structures_sn_40replicas/config.cfg'
        
PR = 0  # contains promoter
AG = 1  #          active gene
NA = 2  #          non-active gene
HS = 3  #          DNase1 hypersensitivity site
CT = 4  #          CTCF site
HM = 5  #          H3K4me3 site
annotations_file = '~/projects/hic/data/bau2011/annotations.txt'
annotations_file = os.path.expanduser(annotations_file)
annotations = np.loadtxt(annotations_file, skiprows=2, dtype=int)

## one RF contains the HS40 site, an enhancer
## (for the alpha-globin genes, I guess)
HS40_bead = 20

## one bead contains the alpha-globin gene cluster
aglobin_bead = 26 ## according to Bau et al. (2011) SI
# aglobin_bead = 28 ## according to Wikipedia

for cfg in (config_file_GM12878, config_file_K562):
    config = parse_config_file(cfg)
    posterior = make_posterior(config)
    data_set = 'GM12878' if cfg == config_file_GM12878 else 'K562'

    n_structures = int(config['general']['n_structures'])
    n_beads = int(config['general']['n_beads'])
    n_samples = int(config['replica']['n_samples']) + 1

    output_folder = config['general']['output_folder']
    samples = load_sr_samples(output_folder + 'samples/', n_replicas, n_samples, 
                              int(config['replica']['samples_dump_interval']),
                              burnin=burnin)
    ens = np.array([sample.variables['structures'].reshape(-1, n_beads, 3)
                    for sample in samples])

    figures_folder = output_folder + 'analysis/bio_validation/'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    current_annotations = annotations[:, (0,1,2,3,4,5) if data_set == 'GM12878'
                                      else (6,7,8,9,10,11)]
    PR_beads = np.where(current_annotations[:,PR] == 1)[0]
    AG_beads = np.where(current_annotations[:,AG] == 1)[0]
    NA_beads = np.where(current_annotations[:,NA] == 1)[0]
    HS_beads = np.where(current_annotations[:,HS] == 1)[0]
    CT_beads = np.where(current_annotations[:,CT] == 1)[0]
    HM_beads = np.where(current_annotations[:,HM] == 1)[0]

    ## Bau et al. (2011) measure smaller distances between HS40
    ## and alpha-globin beads in K562 cells, in which alpha-globin
    ## is transcribed. In GM12878, it is repressed and they measure
    ## longer distances

    ens_flat = ens[::10].reshape(-1, n_beads, 3)
    rgs = np.array(map(radius_of_gyration, ens))
    if not True:
        ## use only structures below radius of gyration threshold
        ## This excludes stretched structures with little contacts
        from csb.bio.utils import radius_of_gyration
        rg_threshold = bead_radii.mean() * 8
        ens_flat = ens_flat[rgs < rg_threshold]
    
    HS40_aglobin_distances = np.sqrt(np.sum((ens_flat[:,HS40_bead] - ens_flat[:,aglobin_bead])**2, 1))
    
    ## now estimate density along the fiber
    bead_radii = posterior.priors['nonbonded_prior'].bead_radii
    density_cutoff = bead_radii.mean() * 3
    densities = []
    for i in range(n_beads):
        ds = np.sqrt(np.sum((ens_flat[:,i][:,None] - ens_flat) ** 2, 2))
        densities.append(sum(ds < density_cutoff) / float(len(ens_flat) * n_beads))
    
    ## calculate and distances between alpha-globin bead and all others
    ag_all_ds = np.sqrt(np.sum((ens_flat[:,aglobin_bead][:,None] - ens_flat) ** 2, 2))

    if data_set == 'K562':
        ds_K562 = HS40_aglobin_distances
        ag_all_ds_K562 = ag_all_ds
        densities_K562 = densities
    else:
        ds_GM12878 = HS40_aglobin_distances
        ag_all_ds_GM12878 = ag_all_ds
        densities_GM12878 = densities
    

fig = plt.figure()
ax = fig.add_subplot(221)
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
ax.set_xticks(np.arange(1, n_beads+1, 5))
ax.set_xticklabels([str(x-1) for x in np.arange(1, n_beads+1, 5)])
ax.legend()

ax = fig.add_subplot(223)
ax.plot(densities_K562, label='K562', lw=2, c='r')
ax.plot(densities_GM12878, label='GM12878', lw=2, c='b')
ax.set_xlabel('bead index')
ax.set_ylabel('fraction of beads within ({:.1f} * avg bead radius) units'.format(density_cutoff))
ax.legend()

if save_figures:
    plt.savefig(figures_folder + 'bio_validation.pdf')
else:
    plt.show()
    
