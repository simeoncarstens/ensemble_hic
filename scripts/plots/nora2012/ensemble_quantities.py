import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from csb.bio.utils import rmsd, radius_of_gyration as rog


def calculate_rgs(X):

    return np.array([map(rog, x) for x in X])


def plot_avg_rg_trace(ax, data_file):

    rogs = np.load(data_file, allow_pickle=True)
    
    skip = 5
    scatter_skip = 30 * 5 / skip * 2
    space = np.arange(1, 50001, 20)[::skip]
    ax.plot(space, rogs[1::skip].mean(axis=1), c='black')
    ax.scatter(space[::scatter_skip].repeat(30).reshape(-1,30),
               rogs[1::skip][::scatter_skip],
               alpha=0.2,s=20, color='black')
    
    ax.set_ylabel('radius of gyration r$_g$ [nm]')
    ax.set_xlabel('# of MCMC samples')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


if __name__ == "__main__":

    import sys
    from cPickle import dump
    from ensemble_hic.setup_functions import parse_config_file
    from ensemble_hic.analysis_functions import load_sr_samples

    cfg_file = sys.argv[1]
    output_file = sys.argv[2]

    scale_factor = 53

    settings = parse_config_file(cfg_file)
    n_replicas = int(settings['replica']['n_replicas'])
    n_structures = int(settings['general']['n_structures'])

    samples = load_sr_samples(settings['general']['output_folder'] + 'samples/',
                              n_replicas, 50001, 1000, 0)
    X = np.array([x.variables['structures'].reshape(n_structures, 308, 3)
                  for x in samples]) * scale_factor
    rogs = np.array([map(rog, x) for x in X])

    with open(output_file, "w") as opf:
        dump(rogs, opf)
