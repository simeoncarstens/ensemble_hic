import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering
from csb.bio.utils import rmsd, average_structure, scale_and_fit, wfit
from scipy.spatial.distance import pdist, squareform
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg, write_ensemble, write_VMD_script, write_pymol_script

def make_dir(d):
    import os
    if not os.path.exists(d):
        os.makedirs(d)
        
waffinities = 'hamming'
waffinities = 'wrmsd'

sys.argv = ['',
            #'/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_40structures_sn_109replicas/config.cfg',
            '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_40structures_sn_122replicas/config.cfg',
            #'/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_10structures_sn_98replicas/config.cfg',
            #'/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_it2_10structures_sn_113replicas/config.cfg',
            #'/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_alpha50_40structures_sn_109replicas/config.cfg',
            8,
            ]

config_file = sys.argv[1]
settings = parse_config_file(config_file)
n_beads = int(settings['general']['n_beads'])
output_folder = settings['general']['output_folder']
cfolder = '{}analysis/clustering/{}/'.format(output_folder, waffinities)
affs_settings = np.load('{}settings.pickle'.format(cfolder))
step = affs_settings['step']
samples = load_samples_from_cfg(config_file)[::step]
n_samples = len(samples)
thold = float(sys.argv[2])
ens = np.array([sample.variables['structures'].reshape(-1, n_beads, 3)
                for sample in samples])
ens = ens.reshape(-1, n_beads, 3)

inds = np.arange(len(ens))
np.random.shuffle(inds)
thold = 0.04 #4 #6
clusters = []
clustered = set()

affs = squareform(np.load('{}affs.npy'.format(cfolder)))
start = np.random.choice(inds)
clusters.append([start])
clustered.add(start)

clusters = []
for m, i in enumerate(inds):
    for c in clusters:
        if np.all(affs[i,np.array(c)] < thold):
            c.append(i)
            break
    else:
        clusters.append([i])
    if m % 100 == 0:
        print m
# while len(clustered) < len(affs): #n_samples:
#     cc = clusters[-1]
#     tba = []
#     ntba = []
#     for i in range(n_samples):
#         if not i in clustered:
#             if np.all(affs[tba,i] < thold) and affs[cc[0],i] < thold:
#                 tba.append(i)
#             else:
#                 ntba.append(i)

#     clusters[-1] += tba
#     [clustered.add(x) for x in tba]
#     if len(ntba) > 0:
#         clusters.append([ntba[0]])
#         clustered.add(ntba[0])

print "{} clusters".format(len(clusters))
print sum(map(len, clusters)) == n_samples
for i in range(100):
    (j, k) = np.random.choice(clusters[0], 2, replace=False)
    if affs[j,k] > thold:
        print j, k, affs[j,k]


if True:
    p = make_posterior(parse_config_file(config_file))
    br = p.priors['nonbonded_prior'].forcefield.bead_radii
    make_dir(cfolder + 'clusters/')
    
    sorted_inds = np.argsort(map(len, clusters))[::-1]
    for i, k in enumerate(sorted_inds):
        c = clusters[k]
        print len(c)

        x = ens[c]
        y = average_structure(x)
        w = br**3
        w/= w.sum()
        for ii in range(len(x)):
            R, t = wfit(y,x[ii],w)
            x[ii,...] = np.dot(x[ii], R.T) + t
        
        write_ensemble(x, '{}c{}.pdb'.format(cfolder + 'clusters/', i))
        write_VMD_script('{}c{}.pdb'.format(cfolder + 'clusters/', i), br,
                         '{}c{}.rc'.format(cfolder + 'clusters/', i))

        from csb.bio.utils import distance_matrix
        mdm = np.mean(map(distance_matrix, ens[c]), 0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ms = ax.matshow(mdm)
        ax.set_title('cluster size: {:.3f}%'.format(len(c) / float(len(ens))))
        fig.colorbar(ms)
        plt.savefig('{}dm_c{}.pdf'.format(cfolder + 'clusters/', i))
        plt.close()


if not True:
    dms = [[pdist(ens[x]) for x in y] for y in clusters]
    mean_dms = np.array([np.mean([squareform(x) for x in y], 0) for y in dms])
