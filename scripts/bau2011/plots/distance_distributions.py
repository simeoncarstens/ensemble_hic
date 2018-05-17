import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/plots/'))
from lib import load_K562_with_ref, load_GM12878_with_ref, br
from physical_units import scale_factor

dms_K562, dms_K562_ref = load_K562_with_ref()
dms_GM12878, dms_GM12878_ref = load_GM12878_with_ref()

ag_beads = (26,)
dhs_beads = (18,19,20)

if False:
    ## watch out when checking locations of genes in genome browsers:
    ## genome assembly is important! Bau et al. (2011) use hg18
    polr3k_beads = (5,)

    f = 2.5
    land = np.logical_and
    norm = lambda x: np.linalg.norm(x, axis=1)

    def contact(dms, bead1, bead2):
        return dms[:,bead1,bead2] < (br[bead1] + br[bead2]) * f

    def contact_mbeads(dms, beads1, beads2):
        masks = []
        for b1 in beads1:
            for b2 in beads2:
                masks.append(contact(dms, b1, b2))
        return np.logical_or.reduce(masks)

    ## calculate fraction of structures with three-way contacts between
    ## enhancers, alpha-globin gene and polr3k gene
    fig, ax = plt.subplots()
    for i, dms in enumerate((dms_K562, dms_GM12878)):
        dms = np.array([squareform(x) for x in dms]) * scale_factor
        n_3way = land(land(contact_mbeads(dms, polr3k_beads, ag_beads2),
                           contact_mbeads(dms, dhs_beads2, ag_beads2)),
                           contact_mbeads(dms, dhs_beads2, polr3k_beads))
        print n_3way.sum() / float(len(dms))
        ax.hist(dms[:,ag_beads,dhs_beads[1]] + \
                dms[:,ag_beads,polr3k_beads] + \
                dms[:,polr3k_beads,dhs_beads[1]],
                bins=100, alpha=0.6, label=('K562','GM12878')[i])
        print dms[:,polr3k_beads,ag_beads2].mean(), dms[:,polr3k_beads,ag_beads2].std()
        print dms[:,dhs_beads2[1],ag_beads2].mean(), dms[:,dhs_beads2[1],ag_beads2].std()
        print dms[:,dhs_beads2[1],polr3k_beads].mean(), dms[:,dhs_beads2[1],polr3k_beads].std()
    ax.legend()
    ## Guersoy et al. (2017) find a larger percentage of 3-way contacts in GM12878;
    ## looks like we do, too


def plot_distance_histograms(beads1, beads2, dms1, dms2,
                             histrange, beads1_label, beads2_label,
                             dms1_label, dms2_label, ax, legend, cd=None):
    
    histargs = {'lw': 2, 'histtype': u'step', 'normed': True, 'range': histrange}
    dms1 = np.array([squareform(dm) for dm in dms1])
    dms2 = np.array([squareform(dm) for dm in dms2])
    d_sel1 = dms1[:,beads1, beads2].ravel()
    d_sel2 = dms2[:,beads1, beads2].ravel()
    
    ax.hist(d_sel1, bins=70, edgecolor='blue', label=dms1_label, **histargs)
    ax.hist(d_sel2, bins=70, edgecolor='orange', label=dms2_label,  **histargs)

    ax.set_yticks(())
    ax.set_xlabel('distance between {} and {} [nm]'.format(beads1_label, beads2_label))
    if len(beads1) == len(beads2) == 1:
        ax.axvline(cd, ls='--', c='r', lw=2)
    if legend:
        ax.legend()

if not False:

    plt.rc('font', size=14)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    beads1 = ag_beads
    beads2 = dhs_beads[:1]
    cd = 1.25 * (br[beads1[0]] + br[beads2[0]]) * scale_factor
    plot_distance_histograms(beads1, beads2, dms_K562 * scale_factor,
                             dms_GM12878 * scale_factor,
                             (0, 750), r'$\alpha$-globin', 'HS site #1',
                             'K562', 'GM12878', ax1, True, cd)
    beads2 = dhs_beads[1:2]
    cd = 1.25 * (br[beads1[0]] + br[beads2[0]]) * scale_factor
    plot_distance_histograms(beads1, beads2, dms_K562 * scale_factor,
                             dms_GM12878 * scale_factor,
                             (0, 750), r'$\alpha$-globin', 'HS site #2',
                             'K562', 'GM12878', ax2, False, cd)
    beads2 = dhs_beads[2:3]
    cd = 1.25 * (br[beads1[0]] + br[beads2[0]]) * scale_factor
    plot_distance_histograms(beads1, beads2, dms_K562 * scale_factor,
                             dms_GM12878 * scale_factor,
                             (0, 750), r'$\alpha$-globin', 'HS site #3',
                             'K562', 'GM12878', ax3, False, cd)
    fig.tight_layout()
    plt.show()

if not False:

    plt.rc('font', size=14)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    beads1 = ag_beads
    beads2 = dhs_beads[:1]
    cd = 1.25 * (br[beads1[0]] + br[beads2[0]]) * scale_factor
    plot_distance_histograms(beads1, beads2, dms_K562 * scale_factor,
                             dms_K562_ref * scale_factor,
                             (0, 750), r'$\alpha$-globin', 'HS site #1',
                             'K562', 'null model', ax1, True, cd)
    beads2 = dhs_beads[1:2]
    cd = 1.25 * (br[beads1[0]] + br[beads2[0]]) * scale_factor
    plot_distance_histograms(beads1, beads2, dms_K562 * scale_factor,
                             dms_K562_ref * scale_factor,
                             (0, 750), r'$\alpha$-globin', 'HS site #2',
                             'K562', 'null model', ax2, False, cd)
    beads2 = dhs_beads[2:3]
    cd = 1.25 * (br[beads1[0]] + br[beads2[0]]) * scale_factor
    plot_distance_histograms(beads1, beads2, dms_K562 * scale_factor,
                             dms_K562_ref * scale_factor,
                             (0, 750), r'$\alpha$-globin', 'HS site #3',
                             'K562', 'null model', ax3, False, cd)
    fig.tight_layout()
    plt.show()
    
