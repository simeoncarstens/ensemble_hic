import sys, os, numpy as np
from scipy.spatial.distance import pdist, squareform
from csb.bio.utils import radius_of_gyration
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg, load_samples

sim_path = '/scratch/scarste/ensemble_hic/bau2011/'
br = np.loadtxt(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/bead_radii_fixed.txt'))
rog = lambda x: radius_of_gyration(x, br ** 3)

def load_reference_simulation(burnin=5000):

    reference_path = sim_path + 'gyration_radius_100replicas3_fixed/'
    sref = load_samples(reference_path + 'samples/', 100, 50001, 5000, burnin)
    Xref = np.array([[x.variables['structures'].reshape(-1, 3) for x in y]
                      for y in sref])
    Xrgs = np.array([map(rog, x) for x in Xref])
    Xrgs_means = Xrgs.mean(1)

    return Xref, Xrgs_means    

def load_sim_with_ref(path, ref_burnin=5000):
    
    rog = lambda x: radius_of_gyration(x, br ** 3)
    s = load_samples_from_cfg(path + 'config.cfg')
    X = np.array([x.variables['structures'].reshape(40, -1, 3)
                  for x in s])
    X_rgs_mean = np.mean(map(rog, X.reshape(-1, 70, 3)))
    Xref, Xrgs_means = load_reference_simulation(ref_burnin)
    Xref = Xref[np.where(Xrgs_means < X_rgs_mean)[0][0]]
    dms = np.array([pdist(x) for x in X.reshape(-1, 70, 3)])
    dms_ref = np.array([pdist(x) for x in Xref.reshape(-1, 70, 3)])

    return dms, dms_ref

def load_K562_with_ref(burnin=25000):

    return load_sim_with_ref(sim_path + 'K562_new_smallercd_nosphere_it2_fixed_40structures_sn_130replicas/',
                             burnin)

def load_GM12878_with_ref(burnin=25000):

    return load_sim_with_ref(sim_path + 'GM12878_new_smallercd_nosphere_fixed_40structures_sn_140replicas/',
                             burnin)
