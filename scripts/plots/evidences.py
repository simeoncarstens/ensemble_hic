import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.special import gammaln
from ensemble_hic.setup_functions import make_posterior, parse_config_file
from ensemble_hic.analysis_functions import load_sr_samples

from csb.numeric import log_sum_exp
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/misc/'))
from simlist import simulations

# which = ('1pga_1shf_fwm_poisson_new_it3',
#          )
# which = ('1pga_1shf_fwm_poisson_new_fixed_it3',
#          )
which = ('nora2012_15kbbins_fixed',)


n_structures = simulations[which[0]]['n_structures']
all_logZs = []
for sim in which:
    current = simulations[sim]
    output_dirs = [current['common_path'] + x for x in current['output_dirs']]
    n_structures = current['n_structures']
    
    logZs = []
    data_terms = []
    for x in output_dirs:
        dos = np.load(x + '/analysis/dos.pickle')
        logZs.append(log_sum_exp(-dos.E.sum(1) + dos.s) - \
                     log_sum_exp(-dos.E[:,1] + dos.s))
        a = x.find('replicas')
        b = x[a-4:].find('_')
        n_replicas = int(x[a-4+b+1:a])

        p = np.load(x + '/analysis/wham_params.pickle')
        c = parse_config_file(x + '/config.cfg')
        s = load_sr_samples(x + '/samples/', n_replicas, p['n_samples']+1,
                            int(c['replica']['samples_dump_interval']),
                            p['burnin'])
        sels = np.load(x + '/analysis/wham_sels.pickle')
        s = s[sels[-1]]
        p = make_posterior(parse_config_file(x + '/config.cfg'))
        L = p.likelihoods['ensemble_contacts']
        d = L.forward_model.data_points[:,2]
        f = gammaln(d+1).sum()
        print "mean log-posterior:", np.mean(map(lambda x: p.log_prob(**x.variables), s))
        logZs[-1] -= f + np.log(len(d)) * (not '1pga' in x)
        data_terms.append(np.array(map(lambda x: -L.log_prob(**x.variables), s)).mean() + f)
        print "evidence:", logZs[-1]
    data_terms = np.array(data_terms)    
    
    all_logZs.append(logZs)

all_logZs = np.array(all_logZs)


def plot_evidences(ax):
    def unique(array):
        uniq, index = np.unique(array, return_index=True)
        return uniq[index.argsort()]
        
    n_structures = simulations[which[0]]['n_structures']
        
    mean_logZs = np.array([all_logZs[0, np.array(n_structures) == i].mean()
                           for i in n_structures])
    mean_logZs = unique(mean_logZs)
    mean_logZs = all_logZs[0,:]
    mean_data_terms = np.array([data_terms[np.array(n_structures) == i].mean()
                                for i in n_structures])
    mean_data_terms = unique(mean_data_terms)
    mean_data_terms = data_terms
    #n_structures = unique(n_structures)
    ax.plot(n_structures, mean_logZs, ls='--', marker='o', label='evidence',
            color='black')
    ax2 = ax.twinx()
    
    ax2.plot(n_structures, mean_data_terms, ls='--', marker='s',
             label='data energy', color='gray')
    ax.set_ylabel('log(evidence) / # data points', color='black', fontsize=14)
    ax2.set_ylabel(r'$-\langle$log $L \rangle$', color='gray', fontsize=14)
    ax.set_xlabel('number of states $n$', fontsize=14)
    ax2.set_xticks(())
    ax.set_xticks(n_structures)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(style='sci', scilimits=(0,0),
                        axis='y')
    ax2.ticklabel_format(style='sci', scilimits=(0,0),
                         axis='y')
    ## make inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_ax = inset_axes(ax, width='40%', height='25%', loc=10)
    inset_ax.plot((10, 15, 20, 25, 30, 35, 35, 40, 100), mean_logZs[2:], ls='--', marker='o',
                  label='evidence', color='black')
    inset_ax.set_xticks((10, 20, 30, 40))
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.set_ylim((-7600, -7300))
    inset_ax.set_xlim((11, 42))
    

if False:
    fig, ax = plt.subplots()
    plot_evidences(ax)
    fig.tight_layout()
    plt.show()
    if not not False:
        fig.set_size_inches((5.8, 2.4))
        fig.tight_layout()



def plot_evidences_proteins():

    ## good values for 1PGA / 1SHF
    top_yticks = (151400, 151800)
    top_yticklabels = ('1.514', '1.518')
    top_ylims = (151300, 151900)
    
    top_yticks = (-1600, -1800, -2000)
    top_yticklabels = ('-1.60', '-1.80', '-2.00')
    top_ylims = (-2080, -1580)

    bottom_ylims = (135000, 136000)
    bottom_yticks = (135000, 136000)

    bottom_ylims = (-17900, -17800)
    bottom_yticks = (-17900, -17800)
    bottom_yticklabels = ('-17.9', '-17.8')
    
    xticks = (1,2,3,4,5,10)
    labels = 'dummy'
    legend = False
    align_y = False
    
    if align_y:
        all_logZs -= all_logZs.max(1)[:,None]

    d = .015  # how big to make the diagonal lines in axes coordinates

    def make_top_ax(ax):

        for i, sim in enumerate(which):
            ax.plot(n_structures, all_logZs[i], ls='--', lw=3, marker='o',
                    markersize=10, label=labels[i], color='black')

        ax.set_ylim(*top_ylims)  # outliers only
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labeltop='off')  # don't put tick labels at the top
        ax.set_xticks(xticks)
        ax.set_yticks(top_yticks)
        kwargs = dict(transform=ax.transAxes, color='black', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax.set_yticklabels(top_yticklabels)
        
    def make_bottom_ax(ax):

        for i, sim in enumerate(which):
            ax.plot(n_structures, all_logZs[i], ls='--', lw=3, marker='o',
                    markersize=10, label=labels[i], color='black')

        ax.set_ylim(*bottom_ylims)  # most of the data

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.tick_bottom()
        ax.set_xticks(xticks)
        ax.set_yticks(bottom_yticks)
        kwargs = dict(transform=ax.transAxes, color='black', clip_on=False)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.set_xlabel('number of states $n$')
        #ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_yticklabels(bottom_yticklabels)

    def make_bothaxes(top_ax, bottom_ax, fig):

        make_top_ax(top_ax)
        make_bottom_ax(bottom_ax)

        btop = top_ax.get_position()
        bbottom = bottom_ax.get_position()
        
        ylabel_pos = (top_ax.get_position().x0 - 0.12,
                      (bbottom.y1 + btop.y0) / 2)
        ylabel = fig.text(ylabel_pos[0], ylabel_pos[1],
                          'log(evidence)', va='center',
                          rotation='vertical')

    def make_plot():

        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 22}
        plt.rc('font', **font)
        fig, (top_ax, bottom_ax) = plt.subplots(2, 1, sharex=True)
        make_bothaxes(top_ax, bottom_ax, fig)
        fig.tight_layout()
