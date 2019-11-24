import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from csb.bio.utils import radius_of_gyration as Rg

from isd.DensityMap import DensityMap, WeightedMap, DataSet

def calc_density(xyz, sigma, spacing=2., lower=None, upper=None, weights=None):

    if lower is None: lower = np.floor(xyz.min(0))
    if upper is None: upper = np.ceil(xyz.max(0))

    shape = tuple(np.ceil((upper-lower)/spacing).astype('i').tolist())

    rho = DataSet(np.zeros(shape).flatten())

    if weights is None:
        emmap = DensityMap(*shape, width=sigma, spacing=spacing)
        emmap.sigma = sigma
    else:
        emmap = WeightedMap(*shape, width=sigma, spacing=spacing)
        emmap.weights = weights
        emmap.sigma = weights * 0 + sigma
    
    emmap.ctype.origin = tuple(lower.tolist())
    emmap._X = xyz
    emmap.fill_mock_data(rho)

    return rho.mock_data.reshape(shape)

def overlap(a, b, sigma=0.5):

    lower = np.concatenate([a,b],0).min(0)
    upper = np.concatenate([a,b],0).max(0)
    
    rho_a = calc_density(a, sigma, spacing=0.5*sigma, lower=lower, upper=upper)
    rho_b = calc_density(b, sigma, spacing=0.5*sigma, lower=lower, upper=upper)

    return np.sum(rho_a*rho_b) / np.sum(rho_a**2)**0.5 / np.sum(rho_b**2)**0.5


def calculate_overlaps(full_samples, nointer_samples, prior_rg_samples):
    
    os.chdir('/tmp/')

    sigmas  = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    sigmas  = (1.0,)

    allresults = {}

    for sigma in sigmas:

        results = []

        for X in (full_samples, nointer_samples, prior_rg_samples):

            a = X.reshape(-1, 308, 3)[:,:109]
            b = X.reshape(-1, 308, 3)[:,109:]

            ov = np.array([overlap(a[i], b[i], sigma=sigma) for i in range(len(a))])
            Ra = np.array(map(Rg, a))
            Rb = np.array(map(Rg, b))

            results.append((ov, Ra, Rb))

            print 'done'

        allresults[sigma] = results

    return allresults


def plot_overlap(ax, data_file):

    correlations = np.load(data_file, allow_pickle=True)

    sigma = 1.0

    overlaps_fulldata = correlations[sigma][0][0]
    overlaps_nointer = correlations[sigma][1][0]
    overlaps_prior_rg = correlations[sigma][2][0]

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
    

if __name__ == "__main__":

    import sys
    from cPickle import dump

    full_samples_file = sys.argv[1]
    nointer_samples_file = sys.argv[2]
    prior_rg_samples_file = sys.argv[3]
    output_file = sys.argv[4]

    X_full = np.array([x.variables['structures']
                               for x in np.load(full_samples_file)])
    X_nointer = np.array([x.variables['structures']
                               for x in np.load(nointer_samples_file)])
    # X_prior_rg = np.array([x.variables['structures']
    #                            for x in np.load(prior_rg_samples_file)])
    X_prior_rg = np.load(prior_rg_samples_file)
    results = calculate_overlaps(X_full, X_nointer, X_prior_rg)

    with open(output_file, "w") as opf:
        dump(results, opf)
