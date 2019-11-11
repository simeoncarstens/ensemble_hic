import sys
import os
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n_beads = 308
    

def plot_md_d_scatter(ax, plot_data_file):

    all_but_sel_x, all_but_sel_y, sels_x, sels_y, maks = np.load(plot_data_file)
    
    settings = samplescfg["settings"]
    samples = samplescfg["samples"]
    
    ax.set_aspect('equal')
    markers = ('*', 's', 'o')
    cs = ('k', 'gray', 'lightgray')

    ax.scatter(all_but_sels_x, all_but_sels_y,
               s=12, marker=markers[-1], c=cs[-1], alpha=0.6, label='$|i-j|>3$')
    for i, (sel_x, sel_y) in enumerate(zip(sels_x, sels_y)):
        ax.scatter(sel_x, sel_y, s=12, marker=markers[i], c=cs[i], alpha=0.6,
                   label='$|i-j|={}$'.format(i+2))
    ax.plot((0, maks * 1.1), (0, maks * 1.1), ls='--', c='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, maks * 1.1)
    ax.set_ylim(0.9, maks * 1.1)
    ax.set_xlabel('experimental counts', fontsize=14)
    ax.set_ylabel('back-calculated counts', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles = (handles[1], handles[2], handles[0])
    labels = (labels[1], labels[2], labels[0])
    ax.legend(handles, labels, frameon=False, title='linear distance [beads]:')

    
if __name__ == "__main__":

    import sys

    cfg_file = sys.argv[1]
    out_file = sys.argv[2]
    
    settings = parse_config_file(cfg_file)
    samples = load_sr_samples(settings['general']['output_folder'] + 'samples/',
                              int(settings['replica']['n_replicas']),
                              50001,
                              1000,
                              30000)
    p = make_posterior(settings)
    fwm = p.likelihoods['ensemble_contacts'].forward_model
    energies = np.array(map(lambda x: -p.log_prob(**x.variables), samples))
    map_sample = samples[np.argmin(energies)]
    n_structures = fwm.n_structures
    sels = [np.where(fwm.data_points[:,1] - fwm.data_points[:,0] == sep)[0]
            for sep in (2,3)]
    antisel = np.array([i for i in np.arange(len(fwm.data_points))
                        if not i in [x for y in sels for x in y]])
    all_but_sel_x = fwm.data_points[antisel,2]
    all_but_sel_y = fwm(**map_sample.variables)[antisel]
    sels_x = [fwm.data_points[sel,2] for sel in sels]
    sels_y = [fwm(**map_sample.variables)[sel] for sel in sels]
    maks = np.concatenate((fwm.data_points[:,2], fwm(**map_sample.variables))).max()
