import sys
import os
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_sr_samples

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/misc/'))
from simlist import simulations

sims = simulations['nora2012_15kbbins_fixed']

def plot_correlations(ax):

    correlations = []
    n_structures = sims['n_structures']
    cpath = sims['common_path']
    opdirs = sims['output_dirs']

    for i, n in enumerate(n_structures):
        sim_path = cpath + opdirs[i] + '/'
        config_file = sim_path + 'config.cfg'
        settings = parse_config_file(config_file)
        # samples = load_sr_samples(sim_path + 'samples/',
        #                           int(settings['replica']['n_replicas']),
        #                           48001, 1000, 30000)

        from ensemble_hic.analysis_functions import load_samples_from_cfg_auto
        samples = load_samples_from_cfg_auto(config_file, burnin=30000)
        
        p = make_posterior(settings)
        fwm = p.likelihoods['ensemble_contacts'].forward_model
        dps = fwm.data_points
        inds = np.where(np.abs(dps[:,0] - dps[:,1]) > 8)
        corrs = []
        for sample in samples:
            md = fwm(**sample.variables)
            corrs.append(np.corrcoef(md[inds], dps[:,2][inds])[0,1])
        correlations.append([np.mean(corrs), np.std(corrs)])

        # energies = np.array(map(lambda x: -p.log_prob(**x.variables), samples))
        # map_sample = samples[np.argmin(energies)]
        # md = fwm(**map_sample.variables)
        # correlations.append(np.corrcoef(md[inds], dps[:,2][inds])[0,1])
    correlations = np.array(correlations)

    # ax.plot(n_structures, correlations, marker='o', ls='--', c='k')
    ax.errorbar(n_structures, correlations[:,0], correlations[:,1],
                marker='o', ls='--', c='k')
    ax.set_xticks(n_structures)
    ax.set_xlabel('number of states $n$')
    ax.set_ylabel(r'$\rho$(back-calculated data, experimental data)')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    if not True:
        ## make inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width='40%', height='25%', loc=10)
        inset_ax.plot(n_structures[2:6], correlations[2:6], ls='--', marker='o',
                      color='black')
        inset_ax.set_xticks(n_structures[2:6])
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['right'].set_visible(False)
        inset_ax.set_ylim((0.985, 1.005))
        inset_ax.set_yticks((0.99, 1.0))
        inset_ax.set_xlim((7, 105))

    

if False:
    d = fwm.data_points
    md = fwm(**map_sample.variables)

    sys.path.append(os.path.expanduser('~/projects/hic/py/hicisd2/'))
    from mantel import *
    
    m_d = np.zeros((308, 308))
    m_d[d[:,0], d[:,1]] = d[:,2]
    m_d[d[:,1], d[:,0]] = d[:,2]

    m_md = np.zeros((308, 308))
    m_md[d[:,0], d[:,1]] = md
    m_md[d[:,1], d[:,0]] = md

    m_d[m_d == 0.0] = m_d[m_d > 0.0].mean()
    m_md[m_md == 0.0] = m_md[m_md > 0.0].mean()

    n = 1000
    corr = np.corrcoef(m_d.ravel(), m_md.ravel())[0,1]
    random_corrs = np.array([np.corrcoef(m_d.ravel(),
                                         diagonal_permute(m_md).ravel())[0,1]
                             for _ in range(n)])
    p = np.sum(random_corrs > corr) / float(n)


if not False:
    fig, ax = plt.subplots()
    plot_correlations(ax)

    plt.show()
