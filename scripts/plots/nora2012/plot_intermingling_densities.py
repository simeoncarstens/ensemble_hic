import os
import numpy as np
import matplotlib.pyplot as plt

correlations = np.load(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/nora2012/intermingling4.pkl'))

sigma = 1.0

overlaps_fulldata = correlations[sigma][0][0]
overlaps_nointer = correlations[sigma][1][0]
overlaps_prior = correlations[sigma][2][0]
overlaps_prior_rg = correlations[sigma][3][0]

def plot_overlap(ax):
    n_bins = 20
    kwargs = dict(histtype='stepfilled', alpha=0.6, normed=True, bins=n_bins)
    ax.hist(overlaps_fulldata, label='full data', **kwargs)
    ax.hist(overlaps_nointer, label='no inter\ncontacts', **kwargs)
    ax.hist(overlaps_prior_rg, label='no data', **kwargs)
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_yticks(())
    ax.set_xlabel('density overlap')
    ax.legend(frameon=False)
    
if False:
    fig, ax = plt.subplots()
    plot_overlap(ax)
    plt.show()
