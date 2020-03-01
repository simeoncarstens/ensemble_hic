"""
Reads in single-cell Hi-C data from Stevens et al. (2017)
Download data from 
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80280
then "Samples (42)" -> "GSM2219{497-504}" -> GSM2219xxx_Cell_x_contact_pairs.txt.gz
"""
import numpy as np
from pyliftover import LiftOver
import matplotlib.pyplot as plt
import sys
from csb.bio.utils import distance_matrix
sys.path.append('../../../data/nora2012/')
from make_processed_files import * 

# start of Tsix TAD
coords_min = 100378306
# end of Xist TAD
coords_max = 101298738


def get_schic_contacts(filename):

    all_contacts = np.loadtxt(filename, dtype=str)

    # filter for cis chrX contacts
    contacts = all_contacts[(all_contacts[:,0] == 'chrX') & (all_contacts[:,2] == 'chrX')]
    contacts = contacts[:,(1,3)].astype(int)
    
    # lift over all contacts from mm10 to mm9
    lo = LiftOver('mm10', 'mm9')
    def do_lift(loc):
        lifted_loc = lo.convert_coordinate('chrX', loc)
        if len(lifted_loc) == 1:
            return lifted_loc[0][1]
        elif len(lifted_loc) > 1:
            raise("Non-unique liftover result")
        else:
            print "Locus {} not in mm9 assembly".format(loc)

    lifted_contacts = np.array(zip(map(do_lift, contacts[:,0]),
                                   map(do_lift, contacts[:,1])))

    # keep only contacts in genomic region of interest
    contacts = contacts[(contacts[:,0] >= coords_min) & (contacts[:,1] <= coords_max)]

    return contacts
    
def map_contacts(contacts):

    from make_processed_files import * 

    # map data to 15 kb bins; for now some copy & pasting from make_processed_files.py
    ## male mESCs
    path = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/')
    rev_fragments1, for_fragments1, matrix1 = parse_5C_file(path + 'GSM873934_male-mESCs-E14-replicate-1.matrix.txt')
    rev_fragments2, for_fragments2, matrix2 = parse_5C_file(path + 'GSM873935_male-mESCs-E14-replicate-2.matrix.txt')
    matrix = (matrix1 + matrix2) / 2.0

    ## set bead size in base pairs
    bead_size = 15000
    ## both TADs
    region_start = 100378306
    region_end = 101298738
    region_revs, region_fors, region = extract_region(matrix,
                                                      region_start, region_end)
    bead_lims = calculate_bead_lims(bead_size, region_revs, region_fors)
    n_beads = len(bead_lims)

    H, _, _ = np.histogram2d(*contacts.T, bins=bead_lims)

    bead_contacts = []
    for i in range(len(H)):
        for j in range(i+1, len(H)):
            n_counts = H[i,j]
            if n_counts > 0:
                bead_contacts.append((i, j, n_counts))

    return np.array(bead_contacts)


fname_template = '../../../data/stevens_2017/GSM2219{}_Cell_{}_contact_pairs.txt'
fnames = [fname_template.format(497 + i, i + 1) for i in range(8)]

all_contacts = map(get_schic_contacts, fnames)


if True:
    # map contacts to beads and plot them against contacts in our structures
    all_bead_contacts = map(map_contacts, all_contacts)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    for i, contacts in enumerate(all_bead_contacts):
        axes[i].scatter(*contacts[:,:2].T, alpha=0.5, color="gray")
        axes[i].scatter(*contacts[:,(1,0)].T, alpha=0.5, color="gray")
        axes[i].set_aspect("equal")
        title = "Cell #{}, {} contacts".format(i+1, len(contacts))
        title += "\n{:.2f} contacts / bead".format(len(contacts) / 62.0)
        axes[i].set_title(title)

        if False:
            lowres_samples = np.load('plot_data/samples_lowres.pickle',
                                     allow_pickle=True)
            X = lowres_samples[-1].variables['structures'].reshape(30, -1, 3)
            dms = np.array(map(distance_matrix, X))
            isd_contacts = dms < 1.5
            axes[-1].matshow(isd_contacts.sum(0), origin='lower')
        else:
            axes[-1].set_visible(False)

    fig.tight_layout()
    plt.show()
