import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/data/eser2017/'))
from yeastlib import rDNA_to_left, chrom_lengths, centromeres
from yeastlib import map_chr_pos_to_bead, determine_chr_from_pos
from yeastlib import map_pos_to_bead
land = np.logical_and
lnot = np.logical_not

this_path = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/')
intra = np.loadtxt(this_path + 'AFac_intra.txt', dtype=str)
inter = np.loadtxt(this_path + 'AFac_inter.txt', dtype=str)

intra = np.array([[x[0][3:], x[1], x[2][3:], x[3], x[4]]
                  for x in intra]).astype(float).astype(int)
inter = np.array([[x[0][3:], x[1], x[2][3:], x[3], x[4]]
                  for x in inter]).astype(float).astype(int)


cl_cumsums = np.cumsum(chrom_lengths[:,1])
centromeres[1:,:] += cl_cumsums[:-1][:,None]

cont_intra = intra.copy()
cont_inter = inter.copy()

all_ifs = np.concatenate((cont_intra, cont_inter))

for i in range(2, len(chrom_lengths) + 1):
        all_ifs[all_ifs[:,0] == i, 1] += cl_cumsums[i-2]
        all_ifs[all_ifs[:,2] == i, 3] += cl_cumsums[i-2]

bin_size = 10000
n_beads = np.ceil((chrom_lengths[:,1].astype(float)) / bin_size).astype(int)
bead_lims = [np.linspace(0, cl, n_beads[i-1]+1, endpoint=False)
             for i, cl in chrom_lengths]
cont_bead_lims = bead_lims[:1] + [bead_lims[i] + cl_cumsums[i-1]
                                  for i in range(1, len(chrom_lengths))]
chrom_ranges = np.insert(np.cumsum(n_beads), 0, 0)

m = np.ones((chrom_ranges[-1], chrom_ranges[-1]))
## fill m block by block in order to avoid shared beads at chromosome boundaries
single_chrom_matrices = []
for i in range(len(chrom_lengths)):
    for j in range(len(chrom_lengths)):
        print i, j
        subifs = all_ifs[land(all_ifs[:,0] == i+1,
                              all_ifs[:,2] == j+1)]
        ## this produces NaN for entries without measurement
        ## but is very slow
        subm = np.zeros((n_beads[i], n_beads[j])) * np.nan
        for k in range(n_beads[i]):
            for l in range(n_beads[j]):
                c1 = cont_bead_lims[i][k] <= subifs[:,1]
                c2 = subifs[:,1] < cont_bead_lims[i][k+1]
                c3 = cont_bead_lims[j][l] <= subifs[:,3]
                c4 = subifs[:,3] < cont_bead_lims[j][l+1]
                a = np.where(land(land(c1, c2), land(c3, c4)))
                if len(a[0]) > 0:
                    subm[k,l] = subifs[a[0],4].sum()

        m[chrom_ranges[i]:chrom_ranges[i+1],
          chrom_ranges[j]:chrom_ranges[j+1]] = subm

        if i == j:
            single_chrom_matrices.append(subm)
m[np.tril_indices(len(m))] = m.T[np.tril_indices(len(m))]
    
if True:
    plotm = m.copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plotm[np.isnan(plotm)] = 0
    ax.matshow(np.log(plotm+1))
    for last_bead in np.cumsum(n_beads):
        ax.plot([-0.5,len(m)],
                [last_bead - 0.5, last_bead - 0.5], color='r', linewidth=0.5)
        ax.plot([last_bead - 0.5, last_bead - 0.5],
                [-0.5,len(plotm) - 0.5], color='r', linewidth=0.5)
    for b, e in centromeres / bin_size:
        for a in (b, e):
            ax.plot([-0.5,len(m)],
                    [a - 0.5, a - 0.5], color='y', linewidth=0.5)
            ax.plot([a - 0.5, a - 0.5],
                    [-0.5,len(plotm) - 0.5], color='y', linewidth=0.5)
    ax.set_xlim((-0.5, len(plotm) - 0.5))
    ax.set_ylim((-0.5, len(plotm) - 0.5))

    fig.tight_layout()
    plt.show()


n_rDNA_beads = 200

if not True:
    chrom = 4
    from yeastlib import write_single_chr_data

    fname = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/chr{}.txt')
    write_single_chr_data(single_chrom_matrices[chrom - 1], fname)

if not True:
    from yeastlib import write_whole_genome_data

    n_rDNA_beads = 150
    fname = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/whole_genome_rDNA{}.txt'.format(n_rDNA_beads))
    write_whole_genome_data(matrix=m, n_rDNA_beads=150,
			    n_beads=n_beads, bead_lims=bead_lims,
			    fname=fname)
    fname = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/mol_ranges_whole_genome_rDNA{}.txt'.format(n_rDNA_beads))
    n_beads_w_rDNA = n_beads.copy()
    n_beads_w_rDNA[11] += n_rDNA_beads
    np.savetxt(fname, np.insert(np.cumsum(n_beads_w_rDNA), 0, 0), fmt='%i')
