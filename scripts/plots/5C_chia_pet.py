import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from csb.bio.utils import rmsd, radius_of_gyration

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg

n_structures = 40

path = '/scratch/scarste/ensemble_hic/bau2011/'
K562_path = path + 'K562_new_smallercd_nosphere_{}structures_sn_109replicas/'.format(n_structures)
GM12878_path = path + 'GM12878_new_smallercd_nosphere_30structures_sn_122replicas/'.format(n_structures)

pK562 = make_posterior(parse_config_file(K562_path + 'config.cfg'))
pGM12878 = make_posterior(parse_config_file(GM12878_path + 'config.cfg'))

sK562 = load_samples_from_cfg(K562_path + 'config.cfg')
sGM12878 = load_samples_from_cfg(GM12878_path + 'config.cfg')

XK562 = np.array([x.variables['structures'].reshape(n_structures, -1, 3)
                  for x in sK562])
XGM12878 = np.array([x.variables['structures'].reshape(30+0*n_structures, -1, 3)
                  for x in sGM12878])

## from the SI of Guersoy et al. (NAR, 2017)
## s: start of locus, e: end of locus
##                      loc1s   loc1e  loc2s   loc2e
chia_pet_ias = array([[ 55200,  56599, 169200, 171999],
                      [ 55000,  56999, 351800, 354999],
                      [ 96800,  98199, 124000, 130599],
                      [ 96600,  98599, 132000, 135599],
                      [ 96600,  98599, 167200, 168799],
                      [ 94200,  96199, 169200, 172199],
                      [ 96800,  98199, 169200, 171999],
                      [ 94897,  95586, 170079, 171864],
                      [ 96798,  97306, 170079, 171864],
                      [ 96600,  98599, 169200, 172199],
                      [ 96600,  98599, 169200, 173399],
                      [ 96600,  98599, 167200, 172199],
                      [100400, 101199, 167200, 168799]])

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/'))
from process_5C_data import make_data, make_beads_RF_list
data_file = os.path.expanduser('~/projects/ensemble_hic/data/bau2011/K562.txt')
_, edm = make_data(data_file)
beads_RFs = np.concatenate(make_beads_RF_list(edm))

def map_pos_to_bead(pos):
    land = np.logical_and
    return beads_RFs[land(beads_RFs[:,1] <= pos, pos < beads_RFs[:,2]),0]

chia_pet_ia_beads = np.array([map_pos_to_bead(x) for x in chia_pet_ias.ravel()])
chia_pet_ia_beads = chia_pet_ia_beads.reshape(chia_pet_ias.shape)

## watch out when checking locations of genes in genome browsers:
## genome assembly is important! Bau et al. (2011) use hg18
polr3k_loc = (36979, 43632)
polr3k_beads = np.array(map(map_pos_to_bead, polr3k_loc)).ravel()
#polr3k_beads = 8
dhs_beads = (19, 20, 21)
#dhs_beads = 21
ag_beads = (26,)
#ag_beads = 26


br = pK562.priors['nonbonded_prior'].forcefield.bead_radii
f = 2.5
land = np.logical_and
norm = lambda x: np.linalg.norm(x, axis=1)

def contact(X, bead1, bead2):
    return norm(X[:,bead1] - X[:,bead2],) < (br[bead1] + br[bead2]) * f

def contact_mbeads(X, beads1, beads2):
    masks = []
    for b1 in beads1:
        for b2 in beads2:
            masks.append(contact(X, b1, b2))
    return np.logical_or.reduce(masks)

## calculate fraction of structures with three-way contacts between
## enhancers, alpha-globin gene and polr3k gene
for X in (XK562, XGM12878):
    X = X.reshape(-1, 70, 3)
    n_3way = land(land(contact_mbeads(X, polr3k_beads, ag_beads),
                       contact_mbeads(X, dhs_beads, ag_beads)),
                       contact_mbeads(X, dhs_beads, polr3k_beads))
    print n_3way.sum() / float(len(X))
## Guersoy et al. (2017) find a larger percentage of 3-way contacts in GM12878;
## looks like we do, too

## calculate fraction of structures with contacts between enhancers
## and alpha-globin gene
for X in (XK562, XGM12878):
    X = X.reshape(-1, 70, 3)
    n_dhs_ag = contact_mbeads(X, dhs_beads, ag_beads)
    print n_dhs_ag.sum() / float(len(X))

def plot_distance_histograms(beads1, beads2, structures, null_structures, ax):
    
    from scipy.spatial.distance import pdist, squareform

    histargs = {'lw': 2, 'histtype': u'step', 'normed': True}
    dms = np.array([squareform(pdist(x)) for x in structures])
    d_sel = dms[:,beads1, beads2].ravel()
    non_sel = [(x, y) for x in range(70) for y in range(x+1, 70)
               if not ((x in beads1 and y in beads2)
                       or (x in beads2 and y in beads1))]
    non_sel = np.array(non_sel)
    d_nonsel = dms[:,non_sel[:,0], non_sel[:,1]].ravel()

    ax.hist(d_sel, bins=50, label='{} - {} (model)'.format(beads1_label,
                                                           beads2_label),
            **histargs)
    ax.hist(d_nonsel, bins=150, label='other distances (model)', **histargs)
    ax.hist(d_nonsel, bins=150, label='{} - {} (null)'.format(beads1_label,
                                                              beads2_label),
            **histargs)
    ax.legend()
