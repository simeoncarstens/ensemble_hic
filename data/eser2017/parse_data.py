import numpy as np
import os
import matplotlib.pyplot as plt
land = np.logical_and
lnot = np.logical_not

this_path = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/')
intra = np.loadtxt(this_path + 'AFac_intra.txt', dtype=str)
inter = np.loadtxt(this_path + 'AFac_inter.txt', dtype=str)

intra = np.array([[x[0][3:], x[1], x[2][3:], x[3], x[4]]
                  for x in intra]).astype(float).astype(int)
inter = np.array([[x[0][3:], x[1], x[2][3:], x[3], x[4]]
                  for x in inter]).astype(float).astype(int)

rDNA_length = 1500000
rDNA_to_left = 450000
## from USCS Genome Browser, SacCer3 April 2011 assembly
chrom_lengths = array([[      1,  230218],
                       [      2,  813184],
                       [      3,  316620],
                       [      4, 1531933],
                       [      5,  576874],
                       [      6,  270161],
                       [      7, 1090940],
                       [      8,  562643],
                       [      9,  439888],
                       [     10,  745751],
                       [     11,  666816],
                       [     12, 1078177 + rDNA_length],
                       [     13,  924431],
                       [     14,  784333],
                       [     15, 1091291],
                       [     16,  948066]])

centromeres = np.array([
    [151465,151582],
    [238207,238323],
    [114385,114501],  
    [449711,449821],  
    [151987,152104],  
    [148510,148627],  
    [496920,497038],  
    [105586,105703],  
    [355629,355745],  
    [436307,436425],  
    [440129,440246],  
    [150828,150947],  
    [268031,268149],  
    [628758,628875],  
    [326584,326702],  
    [555957,556073]  
    ])

cl_cumsums = np.cumsum(chrom_lengths[:,1])
centromeres[1:,:] += cl_cumsums[:-1][:,None]

cont_intra = intra.copy()
cont_inter = inter.copy()

all_ifs = np.concatenate((cont_intra, cont_inter))

for i in range(2, 17):
    if not i == 12:
        all_ifs[all_ifs[:,0] == i, 1] += cl_cumsums[i-2]
        all_ifs[all_ifs[:,2] == i, 3] += cl_cumsums[i-2]
    else:
        c1 = all_ifs[:,0] == i
        c2 = all_ifs[:,2] == i
        c3 = all_ifs[:,1] < rDNA_to_left
        c4 = all_ifs[:,3] < rDNA_to_left
        all_ifs[land(c1, c3),1] += cl_cumsums[i-2]
        all_ifs[land(c2, c4),3] += cl_cumsums[i-2]
        all_ifs[land(c1, lnot(c3)),1] += rDNA_length + cl_cumsums[i-2]
        all_ifs[land(c2, lnot(c4)),3] += rDNA_length + cl_cumsums[i-2]
        
    # else:
    #     cont_intra[cont_intra[:,0] == i, 1] += cl_cumsums[i-2]
    #     cont_intra[cont_intra[:,2] == i, 3] += cl_cumsums[i-2]
    #     cont_inter[cont_inter[:,0] == i, 1] += cl_cumsums[i-2]
    #     cont_inter[cont_inter[:,2] == i, 3] += cl_cumsums[i-2]
        

## correct for rDNA locus
# all_ifs[all_ifs[:,1] >= cl_cumsums[10] + rDNA_to_left, 1] += rDNA_length
# all_ifs[all_ifs[:,3] >= cl_cumsums[10] + rDNA_to_left, 3] += rDNA_length

with_nans = False
if True:
    bin_size = 10000
    n_beads = np.ceil((chrom_lengths[:,1].astype(float)) / bin_size).astype(int)
    # bead_lims = [np.linspace(0, cl, n_beads[i-1]+1, endpoint=not False)
    #              for i, cl in chrom_lengths]
    bead_lims = [np.linspace(0, cl, n_beads[i-1]+1, endpoint=False)
                 for i, cl in chrom_lengths]
    cont_bead_lims = bead_lims[:1] + [bead_lims[i] + cl_cumsums[i-1]
                                      for i in range(1, len(chrom_lengths))]
    chrom_ranges = np.insert(np.cumsum(n_beads), 0, 0)
    
    m = np.ones((chrom_ranges[-1], chrom_ranges[-1]))
    ## fill m block by block in order to avoid shared beads at chromosome boundaries
    for i in range(len(chrom_lengths)):
        for j in range(len(chrom_lengths)):
            print i, j
            subifs = all_ifs[land(all_ifs[:,0] == i+1,
                                  all_ifs[:,2] == j+1)]
            if with_nans:
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
            else:
                ## this creates zero entries where there
                ## may have not been a measurement
                subm, _, _ = np.histogram2d(subifs[:,1], subifs[:,3],
                                            bins=[cont_bead_lims[i],
                                                  cont_bead_lims[j]],
                                            weights=subifs[:,4])

            m[chrom_ranges[i]:chrom_ranges[i+1],
              chrom_ranges[j]:chrom_ranges[j+1]] = subm
    if with_nans:
        m[np.tril_indices(len(m))] = m.T[np.tril_indices(len(m))]
    else:
        m += m.T
        m = m.astype(int)
else:
    bin_size = 10000
    all_ifs_binned = all_ifs.copy()
    all_ifs_binned[:,[1,3]] /= bin_size

    max_bin = all_ifs_binned[:,[1,3]].max()
    m = np.zeros((max_bin + 1, max_bin + 1))
    m[all_ifs_binned[:,1], all_ifs_binned[:,3]] = all_ifs_binned[:,4]
    m[np.diag_indices(len(m))] = 0
    m += m.T

if True:
    plotm = m.copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if with_nans:
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

if not True:
    chrom = 4
    lim_low, lim_high = chrom_ranges[chrom-1], chrom_ranges[chrom]
    path = '~/projects/ensemble_hic/data/eser2017/chr{}_withNaNs.txt'.format(chrom)
    with open(os.path.expanduser(path), 'w') as opf:
        for i in range(lim_low, lim_high):
            for j in range(i, lim_high):
                if np.isnan(m[i,j]):
                    continue
                opf.write('{}\t{}\t{}\n'.format(i-lim_low, j-lim_low,
                                                int(m[i,j])))
