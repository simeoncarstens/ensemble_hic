"""
Reads in single-cell Hi-C data from Nagano et al. (2017)
Download data from
https://bitbucket.org/tanaylab/schic2/src/default/
(section "Sequence processing")
You also need the file "GATC.fends" from the same
repository (section "Translating contact maps into genomic coordinates")
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

adj_file = sys.argv[1]
fends_file = sys.argv[2]
out_file = sys.argv[3]

fends = np.loadtxt(fends_file, usecols=(0,2), dtype=int)
min_fend_id = fends[:,0].min()
max_fend_id = fends[:,0].max()

try:
    fends_contacts = np.loadtxt(adj_file, skiprows=1, dtype=int)
except:
    print adj_file
    sys.exit(0)
conds_0 = (fends_contacts[:,0] >= min_fend_id) & (fends_contacts[:,0] <= max_fend_id)
conds_1 = (fends_contacts[:,1] >= min_fend_id) & (fends_contacts[:,1] <= max_fend_id)
fends_contacts = fends_contacts[conds_0 & conds_1]
mapping = dict(fends)

contacts = np.array([(mapping[i], mapping[j], count)
                     for (i, j, count) in fends_contacts])
if False:
    # start of Tsix TAD
    coords_min = 100378306
    # end of Xist TAD
    coords_max = 101298738

    region_conds0 = (contacts[:,0] >= coords_min) & (contacts[:,0] <= coords_max)
    region_conds1 = (contacts[:,1] >= coords_min) & (contacts[:,1] <= coords_max)
    region_contacts = contacts[region_conds0 & region_conds1]


if False:
    min_bead_separation = 5
    long_range = np.abs(contacts[:,0] - contacts[:,1]) >= min_bead_separation * 15000
    min_one_dist = np.abs(contacts[:,0] - contacts[:,1]) > 15000
    print "{}\t{}\t{}".format(adj_file,
                              len(contacts[min_one_dist]),
                              len(contacts[long_range]))

if not False:
    # map genomic coordinates to the same beads as used in Nora et al. (2012) data
    # and write them out
    
    sys.path.append('../nora2012/')
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
        
        H, _, _ = np.histogram2d(*contacts[:,:2].T, bins=bead_lims)

        bead_contacts = []
        for i in range(len(H)):
            for j in range(len(H)):
                if i != j:
                    n_counts = H[i,j]
                    if n_counts > 0:
                        bead_contacts.append((i, j, n_counts))

        return np.array(bead_contacts)

    print len(contacts)
    bead_contacts = map_contacts(contacts)
    np.savetxt(out_file, bead_contacts, delimiter="\t", fmt="%i")

    

if False:
    np.savetxt(out_file, contacts, delimiter="\t", fmt="%i")

if False:
    scatter_args = dict(color="gray", alpha=0.6)
    
    fig, ax = plt.subplots()

    ax.scatter(*contacts[:,:2].T, **scatter_args)
    ax.scatter(*contacts[:,(1,0)].T, **scatter_args)
    ax.set_aspect("equal")
    ax.set_xlabel("genomic coordinate [bp]")
    ax.set_ylabel("genomic coordinate [bp]")

    plt.show()
