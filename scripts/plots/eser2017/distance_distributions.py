import numpy as np
import os, sys
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import make_posterior, parse_config_file
from ensemble_hic.analysis_functions import load_sr_samples

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/data/eser2017/'))
from yeastlib import CGRep
from model_from_arbona2017 import *
# from cg.utils import rdf

n_structures = 50
n_replicas = 871
suffix = '_kve500'
n_samples = 575
dump_interval = 25
burnin = 300
n_dms = 200
sim_path = '/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017{}_{}structures_sn_{}replicas/'.format(suffix, n_structures, n_replicas)

p = make_posterior(parse_config_file(sim_path + 'config.cfg'))
n_beads = p.priors['nonbonded_prior'].forcefield.n_beads

samples = load_sr_samples(sim_path + 'samples/', n_replicas, n_samples,
                          dump_interval, burnin)
X = np.array([x.variables['structures'].reshape(-1, n_beads, 3) for x in samples])
Xflat = X.reshape(-1, n_beads, 3)

samples_null = load_sr_samples('/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA150_prior2_arbona2017_fixed_1structures_s_100replicas/samples/', 100, 23301, 100, 10000, 5)
Xnull = np.array([x.variables['structures'].reshape(-1, n_beads, 3) for x in samples_null])
Xflatnull = Xnull.reshape(-1, n_beads, 3)

r = CGRep()
cm_beads = r.centromere_beads
if True:
    cm_beads = np.array(list(set(cm_beads.ravel())))
non_cm_beads = np.array([x for x in arange(n_beads) if not x in cm_beads])
tm_beads = r.telomere_beads
non_tm_beads = np.array([x for x in arange(n_beads) if not x in tm_beads])

cm_bead_pairs = np.array([(cm_beads[i], cm_beads[j])
                          for i in range(len(cm_beads))
                          for j in range(i+1, len(cm_beads))])

dms = np.array([squareform(pdist(x)) for x in Xflat[np.random.choice(arange(len(Xflat)), n_dms)]])
#inter_cm_d = dms[:,cm_bead_pairs[:,0], cm_bead_pairs[:,1]]
inter_cm_d = np.array([np.linalg.norm(Xflat[:,pair[0]] - Xflat[:,pair[1]], axis=1)
                       for pair in cm_bead_pairs])
non_cm_bead_pairs = np.array([(i, j)
                              for i in range(n_beads)
                              for j in range(i+1, n_beads)
                              if not (i in cm_beads and j in cm_beads)])
non_cm_d = dms[:,non_cm_bead_pairs[:,0],non_cm_bead_pairs[:,1]]
dmsnull = np.array([squareform(pdist(x)) for x in Xflatnull[np.random.choice(arange(len(Xflatnull)), n_dms)]])
inter_cm_dnull = np.array([np.linalg.norm(Xflatnull[:,pair[0]] - Xflatnull[:,pair[1]], axis=1)
                       for pair in cm_bead_pairs])

scale = our_R_beads * 2

histargs = {'lw': 2, 'histtype': u'step'}
def plot_cm_distances(ax):
    ## distance between centromeric beads
    ax.hist(inter_cm_d.ravel() * scale, bins=50, normed=True,
            edgecolor='blue', label='CM - CM', **histargs)
    ax.hist(non_cm_d.ravel() * scale, bins=50, normed=True,
            edgecolor='orange', label='non CM - non CM',**histargs)
    ax.hist(inter_cm_dnull.ravel() * scale, bins=50, normed=True,
            edgecolor='green', label='CM - CM (null)',**histargs)
    ax.plot((inter_cm_d.mean() * scale, inter_cm_d.mean() * scale),
            (0, ax.get_ylim()[1]),
            c='blue', ls='--', label='mean CM - CM')
    ax.plot((non_cm_d.mean() * scale, non_cm_d.mean() * scale),
            (0, ax.get_ylim()[1]),
            c='orange', ls='--', label='mean non CM - non CM')
    ax.plot((inter_cm_dnull.mean() * scale, inter_cm_dnull.mean() * scale),
            (0, ax.get_ylim()[1]),
            c='green', ls='--', label='mean CM - CM (null)')
    ax.legend()
    ax.set_xlabel('distance [nm]')
    ax.set_yticks(())
    ax.set_xlim((0, 3000))

def plot_tm_distances(ax):
    ## distance between telomeric beads
    tm_d = np.linalg.norm(Xflat[:,tm_beads], axis=2)
    non_tm_d = np.linalg.norm(Xflat[:,non_tm_beads], axis=2)
    tm_dnull = np.linalg.norm(Xflatnull[:,non_tm_beads], axis=2)
    ax.hist(tm_d.ravel() * scale, bins=50, normed=True,
            edgecolor='blue', label='TEL beads',**histargs)
    ax.hist(non_tm_d.ravel() * scale, bins=50, normed=True,
            edgecolor='orange', label='non-TEL beads',**histargs)
    ax.hist(tm_dnull.ravel() * scale, bins=50, normed=True,
            edgecolor='green', label='TEL beads (null)',**histargs)
    ax.plot((tm_d.mean() * scale, tm_d.mean() * scale),
            (0, ax.get_ylim()[1]),
            c='blue', ls='--', label='mean telomere beads')
    ax.plot((non_tm_d.mean() * scale, non_tm_d.mean() * scale), (0, ax.get_ylim()[1]),
            c='orange', ls='--', label='mean non-telomere beads')
    ax.plot((tm_dnull.mean() * scale, tm_dnull.mean() * scale),
            (0, ax.get_ylim()[1]),
            c='green', ls='--', label='mean telomere beads (null)')
    ax.legend()
    ax.set_xlabel('distance [nm]')
    ax.set_yticks(())
    ax.set_xlim((0, 3000))


from yeastlib import map_chr_pos_to_bead

def plot_chr4_distances(data_file, loc1, loc2, ax):
    f = 1.0
    if False:
        from yeastlib import centromeres
        data = np.loadtxt(data_file)
        ## both loci are on right arm and in units kb from CEN
        bead1 = map_chr_pos_to_bead(4, centromeres[3,1] + loc1, r.bead_lims)
        bead2 = map_chr_pos_to_bead(4, centromeres[3,1] + loc2, r.bead_lims)
        bead1 += r.n_beads.cumsum()[2]
        bead2 += r.n_beads.cumsum()[2]
    else:
        ## turns out the genomic coordinates in the data files are absolute,
        ## that is, not measured from the centromere
        data = np.loadtxt(data_file)
        bead1 = map_chr_pos_to_bead(4, loc1, r.bead_lims)
        bead2 = map_chr_pos_to_bead(4, loc2, r.bead_lims)
        bead1 += r.n_beads.cumsum()[2]
        bead2 += r.n_beads.cumsum()[2]

    d = np.linalg.norm(Xflat[:,bead1] - Xflat[:,bead2], axis=1) * f
    dnull = np.linalg.norm(Xflatnull[:,bead1] - Xflatnull[:,bead2], axis=1) * f
    print bead1, bead2
    tmpstr = '{}kb to\n{} kb  '.format(loc1/1000, loc2/1000)
    ax.hist(d.ravel() * scale, bins=10, normed=True, edgecolor='blue',
            label=tmpstr+'(model)',**histargs)
    ax.hist(data * 1000, bins=40, normed=True, edgecolor='orange',
            label=tmpstr+'(exp)',**histargs)
    ax.hist(dnull.ravel() * scale, bins=50, normed=True, edgecolor='green',
            label=tmpstr+'(null)',**histargs)
    ax.legend()
    ax.set_xlabel('distance [nm]')
    ax.set_yticks(())
    ax.set_xlim((0, 3000))

def plot_telomere_distances(data_file, tel1, tel2, ax):

    f = 1.0
    data = np.loadtxt(data_file)
    n_beads_cs = r.n_beads.cumsum()
    bead1 = n_beads_cs[tel1-2]
    bead2 = n_beads_cs[tel2-1] - 1
    print bead1, bead2
    d = dms[:,bead1,bead2] * f
    dnull = dmsnull[:,bead1,bead2] * f
    tmpstr = 'Tel{}L - Tel{}R '.format(tel1, tel2)
    ax.hist(d.ravel() * scale, bins=20, normed=True, edgecolor='blue',
            label=tmpstr+'(model)', **histargs)
    ax.hist(data * 1000, bins=50, normed=True, edgecolor='orange',
            label=tmpstr+'(exp)', **histargs)
    ax.hist(dnull.ravel() * scale, bins=20, normed=True, edgecolor='green',
            label=tmpstr+'(null)', **histargs) 
    ax.legend()
    ax.set_xlabel('distance [nm]')
    ax.set_yticks(())
    ax.set_xlim((0, 3000))


dpath = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/arbona2017/')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}
plt.rc('font', **font)

fig = plt.figure()

ax = fig.add_subplot(341)
plot_cm_distances(ax)

ax = fig.add_subplot(342)
plot_tm_distances(ax)

ax = fig.add_subplot(343)
plot_telomere_distances(dpath + 'tel12L_tel4R.txt', 12, 4, ax)

ax = fig.add_subplot(344)
plot_telomere_distances(dpath + 'tel6L_tel10R.txt', 6, 10, ax)

ax = fig.add_subplot(345)
plot_telomere_distances(dpath + 'tel6L_tel6R.txt', 6, 6, ax)

ax = fig.add_subplot(346)
plot_telomere_distances(dpath + 'tel3L_tel6R.txt', 3, 6, ax)

ax = fig.add_subplot(347)
plot_chr4_distances(dpath + 'chr4R_chr4R_854kb_976kb.txt',
                    854000, 976000, ax)

ax = fig.add_subplot(348)
plot_chr4_distances(dpath + 'chr4R_chr4R_1145kb_1185kb.txt',
                    1145000, 1185000, ax)

ax = fig.add_subplot(349)
plot_chr4_distances(dpath + 'chr4R_chr4R_1185kb_1018kb.txt',
                    1185000, 1018000, ax)

ax = fig.add_subplot(3,4,10)
plot_chr4_distances(dpath + 'chr4R_chr4R_1185kb_1095kb.txt',
                    1185000, 1095000, ax)

plt.show()

