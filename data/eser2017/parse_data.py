import numpy as np
import os
import matplotlib.pyplot as plt

this_path = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/')
intra = np.loadtxt(this_path + 'AFac_intra.txt', dtype=str)
inter = np.loadtxt(this_path + 'AFac_inter.txt', dtype=str)

intra = np.array([[x[0][3:], x[1], x[2][3:], x[3], x[4]]
                  for x in intra]).astype(float).astype(int)
inter = np.array([[x[0][3:], x[1], x[2][3:], x[3], x[4]]
                  for x in inter]).astype(float).astype(int)

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
                       [     12, 1078177],
                       [     13,  924431],
                       [     14,  784333],
                       [     15, 1091291],
                       [     16,  948066]])
cl_cumsums = np.cumsum(chrom_lengths[:,1])

cont_intra = intra.copy()
cont_inter = inter.copy()
for i in range(2, 17):
    cont_intra[cont_intra[:,0] == i, 1] += cl_cumsums[i-2]
    cont_intra[cont_intra[:,2] == i, 3] += cl_cumsums[i-2]
    cont_inter[cont_inter[:,0] == i, 1] += cl_cumsums[i-2]
    cont_inter[cont_inter[:,2] == i, 3] += cl_cumsums[i-2]

all_ifs = np.concatenate((cont_intra, cont_inter))

bin_size = 10000
all_ifs_binned = all_ifs.copy()
all_ifs_binned[:,[1,3]] /= bin_size

max_bin = all_ifs_binned[:,[1,3]].max()
m = np.zeros((max_bin + 1, max_bin + 1))
m[all_ifs_binned[:,1], all_ifs_binned[:,3]] = all_ifs_binned[:,4]
m[np.diag_indices(len(m))] = 0
m += m.T

fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(m)
for cs in cl_cumsums:
    pos = cs / bin_size
    ax.plot([0,len(m)], [pos, pos], color='r', linewidth=0.5)
    ax.plot([pos, pos], [0,len(m)], color='r', linewidth=0.5)
ax.set_xlim((0, len(m)))
ax.set_ylim((0, len(m)))

fig.tight_layout()
plt.show()
