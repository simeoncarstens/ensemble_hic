import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import spectral_clustering
from csb.bio.utils import rmsd, average_structure, scale_and_fit
from scipy.spatial.distance import pdist, squareform
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg, write_ensemble, write_VMD_script, write_pymol_script

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/clustering/'))
from clustering_funcs import *

sys.argv = ['',
            # '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_30structures_sn_122replicas/config.cfg',
            '/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_alpha50_40structures_sn_109replicas/config.cfg',
            # '/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_40structures_sn_109replicas/config.cfg',            
            1,
            2,
            6,
            1]

config_file = sys.argv[1]
n_processes = int(sys.argv[2])
n_clusters_range = range(int(sys.argv[3]), int(sys.argv[4]))
step = int(sys.argv[5])
settings = parse_config_file(config_file)
output_folder = settings['general']['output_folder']
samples = load_samples_from_cfg(config_file)[::step]

if 'weights' in samples[-1].variables:
    weights = np.array([x.variables['weights'] for x in samples.ravel()])
if 'norm' in samples[-1].variables:
    norms = np.array([x.variables['norm'] for x in samples.ravel()])

ens = np.array([sample.variables['structures'].reshape(-1, 70, 3)
                for sample in samples])
p = make_posterior(settings)
bead_radii = p.priors['nonbonded_prior'].forcefield.bead_radii
L = p.likelihoods['ensemble_contacts']
FWM = L.forward_model
alpha = float(settings['forward_model']['alpha'])

figures_folder = output_folder + 'analysis/clustering/'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

from ensemble_hic.forward_models import EnsembleContactsFWM
FWM = EnsembleContactsFWM('q', 1, FWM['contact_distances'].value,
                          FWM.data_points)
fwm_eval = lambda x: FWM(structures=x[None,:], weights=np.ones(1),
                         norm=1.0, smooth_steepness=alpha)
dps = FWM.data_points

mds = np.array([[fwm_eval(x) for x in y] for y in ens])
mults = np.array([np.dot(x.T, x) for x in mds])



from sklearn.decomposition import NMF
n_components = 2
def nmf(x):
    model = NMF(n_components=n_components,
                init='random', random_state=0, tol=1e-10,
                max_iter=1000)
    model.verbose=0
    W = model.fit_transform(x)
    H = model.components_

    return W, H

res = map(nmf, mults)
ws, hs = np.array([x[0] for x in res]), np.array([x[1] for x in res])
ws /= ws.sum(2)[:,:,None]
    

figure();
[plt.plot(sort(prob[:,i]),linspace(0,1,len(prob),endpoint=False)) for i in range(n_components)]
plt.show()


def rmsd(X, Y):
    """
    Calculate the root mean squared deviation (RMSD) using Kabsch' formula.

    @param X: (n, d) input vector
    @type X: numpy array

    @param Y: (n, d) input vector
    @type Y: numpy array

    @return: rmsd value between the input vectors
    @rtype: float
    """

    from numpy import sum, dot, sqrt, clip, average
    from numpy.linalg import svd, det

    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    R_x = sum(X ** 2)
    R_y = sum(Y ** 2)

    V, L, U = svd(dot(Y.T, X))

    return sqrt(clip(R_x + R_y - 2 * sum(L), 0., 1e300) / len(X))

import hungarian
from munkres import Munkres

m = Munkres()

#assignments = [np.arange(X.shape[1])]
for i in range(len(X)-1):
    r = np.array([[rmsd(x,y) for y in X[i+1]] for x in X[i]])
    if not False:
        j = hungarian.lap(r.copy())[0]
        pairs = zip(range(len(j)),j)
    else:
        pairs = m.compute(r.copy())
    print i, np.sum([r[a,b] for a, b in pairs]), 
    X[i+1] = X[i+1][j]
    print np.sum([rmsd(x,y) for x,y in zip(X[i],X[i+1])])
    
#    assignments.append(j)
