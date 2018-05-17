import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from csb.bio.utils import rmsd, radius_of_gyration
from scipy.spatial.distance import pdist, squareform
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/plots/'))
from lib import load_K562_with_ref, load_GM12878_with_ref

n_structures_K562 = 40
n_structures_GM12878 = 40
dms_K562, dms_K562_ref = load_K562_with_ref()
dms_GM12878, dms_GM12878_ref = load_GM12878_with_ref()


cds = np.array([1.2 * (br[i] + br[j]) for i in range(len(br)) for j in range(i+1, len(br))])

if True:
    cf = lambda dms, cds=cds: dms < cds
else:
    alpha = pK562['smooth_steepness'].value
    cf = lambda dms, cds=cds: 0.5 * (alpha*(cds-d)/np.sqrt(1+alpha**2*(cds-d)**2)) + 0.5

cK562_mean = cf(dms_K562).reshape(-1, n_structures_K562, dms_K562.shape[1]).mean(1).mean(0)
cK562_ref_mean = cf(dms_K562_ref).reshape(-1, dms_K562.shape[1]).mean(0)
cGM12878_mean = cf(dms_GM12878).reshape(-1, n_structures_GM12878, dms_GM12878.shape[1]).mean(1).mean(0)
cGM12878_ref_mean = cf(dms_GM12878_ref).reshape(-1, dms_GM12878.shape[1]).mean(0)


dpath = os.path.expanduser('~/projects/ensemble_hic/data/bau2011/chia_pet/')
sys.path.append(dpath)
from parse import *
import glob
c_heidari = parse_heidari2014(*glob.glob(dpath + 'heidari2014/GM12878/*.txt'))
c_heidari = map_contacts_to_beads(c_heidari, dpath + '../K562.txt')
c_li = parse_li2012(dpath + 'li2012/mmc2.xls')
c_li = map_contacts_to_beads(c_li, dpath + '../K562.txt')

if True:
    c_heidari = filter_not_in_5C(c_heidari, dpath + '../K562.txt')
    c_li = filter_not_in_5C(c_li, dpath + '../K562.txt')
    c_all = np.concatenate((c_heidari, c_li))
    c_condensed = np.vstack((c_all[:,:2].mean(1).astype(int),
                             c_all[:,2:4].mean(1).astype(int))).T
    c_condensed = c_condensed[c_condensed[:,1] - c_condensed[:,0] > 0]
    c_condensed = np.vstack({tuple(row) for row in c_condensed})
    lr = c_condensed[c_condensed[:,1] - c_condensed[:,0] > 10]
    fiveC_contacts = np.loadtxt(dpath + '../K562_processed_fixed.txt').astype(int)[:,:2]
    fiveC_contacts = np.sort(fiveC_contacts[:,:2], 1)
    non_contacts = np.array([(i,j) for i in range(70) for j in range(i+1, 70)
                             if not np.any(np.array([i,j])[None,:] - fiveC_contacts == 0)
                            ])
    
if not True:

    import sys
    sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/plots/'))
    from physical_units import scale_factor

    # dms_GM12878 = np.array([squareform(x) for x in dms_GM12878]).reshape(-1,40,70,70)
    # dms_GM12878_ref = np.array([squareform(x) for x in dms_GM12878_ref])

    dms_K562 = np.array([squareform(x) for x in dms_K562]).reshape(-1,40,70,70)
    dms_K562_ref = np.array([squareform(x) for x in dms_K562_ref])

    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # logratio = np.log(dms_GM12878.reshape(-1,70,70).mean(0) / dms_GM12878_ref.mean(0))
    logratio = (dms_GM12878.reshape(-1,70,70).mean(0) - dms_GM12878_ref.mean(0)) * scale_factor
    ms = ax.matshow(logratio)
    ax.scatter(c_heidari[:,:2].mean(1), c_heidari[:,2:4].mean(1), c='r',
               label='ChIA-PET contacts (not in 5C)', s=6)
    ax.scatter(c_li[:,:2].mean(1), c_li[:,2:4].mean(1), c='y',
               label='ChIA-PET contacts (not in 5C)')
    cb = fig.colorbar(ms, fraction=0.046, pad=0.04)
    cb.set_label('log(mean model distance / mean ref distance)')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.hist(logratio[c_condensed[:,0], c_condensed[:,1]].ravel(),
            bins=int(np.sqrt(700)), alpha=0.6, normed=True,
            label='fraction of ensemble members\nforming ChIA-PET contacts (model)')
    ax.legend()
    fig.tight_layout()
    plt.show()

if True:
    import sys
    sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/plots/'))
    from physical_units import scale_factor

    # dms_GM12878 = np.array([squareform(x) for x in dms_GM12878]).reshape(-1,40,70,70)
    # dms_GM12878_ref = np.array([squareform(x) for x in dms_GM12878_ref])

    # dms_K562 = np.array([squareform(x) for x in dms_K562]).reshape(-1,40,70,70)
    # dms_K562_ref = np.array([squareform(x) for x in dms_K562_ref])

    dms_flat = dms_K562.reshape(-1, 70, 70) * scale_factor
    maks = dms_flat[:,lr[:,0],lr[:,1]].max()
    fig, axes = plt.subplots(8,5,sharex=True,sharey=True)
    axes = axes.ravel()
    for i, (cx, cy) in enumerate(lr):
        axes[i].hist(dms_flat[:,cx,cy], bins=70, range=(0, maks))
        axes[i].axvline(squareform(cds)[cx,cy] * scale_factor,
                        ls='--', c='r', lw=1)
    fig.tight_layout()
    plt.show()

if not True:

    KL = np.load('/usr/users/scarste/test/KL_K562_KDE_log.npy')
    KL2 = squareform(KL)
    
    fig, ax = plt.subplots()
    KL_expcontacts = KL2[lr[:,0], lr[:,1]]
    KL_all = np.array([KL2[i,j] for i in range(70) for j in range(i+1,70)
                       if j-i > 10])
    KL_all = []
    for i in range(1000):
        for x, y in lr:
            KL_all.append(np.random.choice(np.diag(KL2, y-x)))

    nc_hashes = {'{:02d}{:02d}'.format(*x) for x in non_contacts}
    exp_hashes = {'{:02d}{:02d}'.format(*x) for x in lr}
    neither_in_ChIA_nor_5C = nc_hashes.difference(exp_hashes)
    neither_in_ChIA_nor_5C = np.array([(int(x[:2]), int(x[2:]))
                                       for x in neither_in_ChIA_nor_5C])
    KL_all = KL2[neither_in_ChIA_nor_5C[:,0],neither_in_ChIA_nor_5C[:,1]]
    
    ax.hist(KL_expcontacts, bins=5, normed=True, alpha=0.6,
            label='ChIA-PET contacts\n(not in 5C data)')
    ax.hist(KL_all, bins=70, normed=True, alpha=0.6,
            label='all contacts')
    ax.legend()
    plt.show()
