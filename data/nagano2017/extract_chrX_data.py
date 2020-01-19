"""
Reads in single-cell Hi-C data from Nagano et al. (2017)
Download data from
https://bitbucket.org/tanaylab/schic2/src/default/
(section "Sequence processing")
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

adj_file = sys.argv[1]
out_file = sys.argv[2]

fends_chrX = np.loadtxt("GATC_chrX.fends", usecols=(0,2), dtype=int)
min_fend_id = fends_chrX[:,0].min()
max_fend_id = fends_chrX[:,0].max()

fends_contacts = np.loadtxt(adj_file, skiprows=1, dtype=int)
conds_0 = (fends_contacts[:,0] >= min_fend_id) & (fends_contacts[:,0] <= max_fend_id)
conds_1 = (fends_contacts[:,1] >= min_fend_id) & (fends_contacts[:,1] <= max_fend_id)
fends_contacts_chrX = fends_contacts[conds_0 & conds_1]
mapping = dict(fends_chrX)

contacts = np.array([(mapping[i], mapping[j], count)
                     for (i, j, count) in fends_contacts_chrX])

np.savetxt(out_file, contacts, delimiter="\t", fmt="%i")

if True:
    scatter_args = dict(color="gray", alpha=0.6)
    
    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.scatter(*contacts[:,:2].T, **scatter_args)
    ax1.scatter(*contacts[:,(1,0)].T, **scatter_args)
    ax1.set_title("chrX")
    ax1.set_aspect("equal")
    ax1.set_xlabel("genomic coordinate [bp]")
    ax1.set_ylabel("genomic coordinate [bp]")
    
    # start of Tsix TAD
    coords_min = 100378306
    # end of Xist TAD
    coords_max = 101298738

    region_conds0 = (contacts[:,0] >= coords_min) & (contacts[:,0] <= coords_max)
    region_conds1 = (contacts[:,1] >= coords_min) & (contacts[:,1] <= coords_max)
    region_contacts = contacts[region_conds0 & region_conds1]

    ax2.scatter(*region_contacts[:,:2].T, **scatter_args)
    ax2.scatter(*region_contacts[:,(1,0)].T, **scatter_args)
    ax2.set_title("Tsix and Xist TADs")
    ax2.set_aspect("equal")
    ax2.set_xlabel("genomic coordinate [bp]")
    ax2.set_ylabel("genomic coordinate [bp]")

    plt.show()
