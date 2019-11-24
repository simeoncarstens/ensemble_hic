import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial.distance import pdist, squareform
from csb.bio.utils import rmsd, radius_of_gyration as rog
from ensemble_hic.analysis_functions import load_sr_samples


def calculate_rgs(X):
    
    tad1 = X[:,:,:107]
    tad2 = X[:,:,107:]
    tad1flat = tad1.reshape(-1, 107,3)
    tad2flat = tad2.reshape(-1, 201,3)

    rogs_tad1 = np.array(map(rog, tad1flat))
    rogs_tad2 = np.array(map(rog, tad2flat))
 
    return rogs_tad1, rogs_tad2


def plot_rg_hist(ax, data_file):

    rogs_tad1, rogs_tad2 = np.load(data_file, allow_pickle=True)

    tad1_color = 'red'
    tad2_color = 'blue'
    ax.hist(rogs_tad1, bins=100, label='Tsix TAD', alpha=0.6, color=tad1_color,
            histtype='stepfilled', normed=True)
    ax.axvline(rogs_tad1.mean(), ls='--', color=tad1_color, lw=2)
    ax.hist(rogs_tad2, bins=100, label='Xist TAD', alpha=0.6, color=tad2_color,
            histtype='stepfilled', normed=True)
    ax.axvline(rogs_tad2.mean(), ls='--', color=tad2_color, lw=2)
    ax.set_xlim((150, 350))
    ax.set_xlabel(r'radius of gyration $r_g$ [nm]')
    ax.set_yticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


if False:
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 22}
    plt.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    plot_before_hist(ax1)
    plot_after_hist(ax2)
    plt.gcf().tight_layout()
    plt.show()


if __name__ == "__main__":

    import sys
    from cPickle import dump
    from ensemble_hic.setup_functions import parse_config_file

    before_cfg_file = sys.argv[1]
    after_cfg_file = sys.argv[2]
    before_out_file = sys.argv[3]
    after_out_file = sys.argv[4]

    scale_factor = 53

    for cfg_file, out_file in zip((before_cfg_file, after_cfg_file),
                                  (before_out_file, after_out_file)):

        settings = parse_config_file(cfg_file)
        n_replicas = int(settings['replica']['n_replicas'])
        n_structures = int(settings['general']['n_structures'])
        samples = load_sr_samples(settings['general']['output_folder'] + 'samples/',
                              n_replicas, 45001, 1000, 25000)
        X = np.array([x.variables['structures'].reshape(n_structures, 308, 3)
                      for x in samples]) * scale_factor
        rgs = calculate_rgs(X)

        with open(out_file, "w") as opf:
            dump(rgs, opf)
