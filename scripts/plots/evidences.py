import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from csb.numeric import log_sum_exp
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/misc/'))
from simlist import simulations

# which = ('GM12878_new_smallercd_nosphere',
#          'K562_new_smallercd_nosphere',
#         )

# which = ('rao2014_randominit',
#          )

# which = ('GM12878_new_smallercd_nosphere_alpha50',
#          'K562_new_smallercd_nosphere_alpha50',
#         )
# which = ('eser2017_chr4',
#         )
# which = ('1pga_1shf_fwm_poisson_new_it3',
#          )

# which = ('eser2017_whole_genome',
#          )
# which = ('GM12878_new_smallercd_nosphere_fixed',
#          'K562_new_smallercd_nosphere_fixed',
#         )
# which = ('K562_new_smallercd_nosphere_fixed',
#          'GM12878_new_smallercd_nosphere_fixed',
#          )
# which = ('hairpin_s_fwm_poisson_new',)
which = ('nora2012',)

all_logZs = []
for sim in which:
    current = simulations[sim]
    output_dirs = [current['common_path'] + x for x in current['output_dirs']]
    n_structures = current['n_structures']
    
    logZs = []
    data_terms = []
    entropy_terms = []
    bla = []
    for x in output_dirs:
        dos = np.load(x + '/analysis/dos.pickle')
        logZs.append(log_sum_exp(-dos.E.sum(1) + dos.s) - \
                     log_sum_exp(-dos.E[:,1] + dos.s))
        print logZs[-1]
        print dos.log_Z(1)
        data_terms.append(log_sum_exp(-dos.E.sum(1) + dos.s))
        entropy_terms.append(log_sum_exp(-dos.E[:,1] + dos.s))
    data_terms = np.array(data_terms)    
    entropy_terms = np.array(entropy_terms)    

    all_logZs.append(logZs)

all_logZs = np.array(all_logZs)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

if not True:
    fig, (top_ax, bottom_ax) = plt.subplots(2, 1, sharex=True)
    if True:
        ## good values for K562 and GM12878
        ## have to move 1e4 axis label on top in Inkscape
        top_ylims = (-500, 100)
        bottom_ylims = (-32000, -28000)
        xticks = (1,7,10,15,20,30,40,50)
        top_yticks = (0, -500)
        top_yticklabels = ('0', '-0.05')
        bottom_yticks = (-28000, -32000)
        labels = ('K562', 'GM12878')
        legend = True
        align_y = True

    if True:
        ## good values for 1PGA / 1SHF
        top_yticks = (151400, 151800)
        top_yticklabels = ('1.514', '1.518')
        top_ylims = (151300, 151800)
        bottom_ylims = (135000, 136000)
        bottom_yticks = (135000, 136000)
        xticks = (1,2,3,4,5,10)
        labels = 'dummy'
        legend = False
        align_y = False

    if not True:
        ## good values for hairpin / S
        top_ylims = (12270, 12530)
        bottom_ylims = (10600, 10700)
        xticks = (1,2,3,4,5,10)
        top_yticks = (12200, 12500)
        top_yticklabels = ('1.22', '1.25')
        bottom_yticks = (10640, 10680)
        labels = 'dummy'
        legend = False
        align_y = False

    if align_y:
        all_logZs -= all_logZs.max(1)[:,None]

    for i, sim in enumerate(which):
        top_ax.plot(n_structures, all_logZs[i], ls='--', marker='o', label=labels[i])
        bottom_ax.plot(n_structures, all_logZs[i], ls='--', marker='o', label=labels[i])

    top_ax.set_ylim(*top_ylims)  # outliers only
    bottom_ax.set_ylim(*bottom_ylims)  # most of the data

    top_ax.spines['bottom'].set_visible(False)
    top_ax.spines['top'].set_visible(False)
    bottom_ax.spines['top'].set_visible(False)
    top_ax.xaxis.set_visible(False)
    top_ax.spines['right'].set_visible(False)
    bottom_ax.spines['right'].set_visible(False)
    top_ax.tick_params(labeltop='off')  # don't put tick labels at the top
    bottom_ax.xaxis.tick_bottom()
    top_ax.set_xticks(xticks)
    bottom_ax.set_xticks(xticks)
    top_ax.set_yticks(top_yticks)
    bottom_ax.set_yticks(bottom_yticks)

    ## plot cut axes symbol
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=top_ax.transAxes, color='k', clip_on=False)
    top_ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #top_ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=bottom_ax.transAxes)  # switch to the bottom axes
    bottom_ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #bottom_ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    fig.text(0.0, 0.5, 'log-evidence', va='center', rotation='vertical')
    bottom_ax.set_xlabel('# of states')
    top_ax.set_yticklabels(top_yticklabels)
    from matplotlib.ticker import ScalarFormatter
    # top_ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1E}'))
    # bottom_ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1E}'))
    #top_ax.yaxis.set_major_formatter(ScalarFormatter())
    bottom_ax.yaxis.set_major_formatter(ScalarFormatter())

    if legend:
        top_ax.legend()
    #top_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    bottom_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
else:
    fig, ax = plt.subplots()
    ax.plot(n_structures, all_logZs[0], ls='--', marker='o', label='evidence')
    ax.plot(n_structures, data_terms, ls='--', marker='o', label='likelihood\ncontribution')
    ax.set_ylabel('log-evidence /\n-likelihood contribution')
    ax.set_xlabel('# of states')
    ax.set_yticks(np.array([32,34,36,38]) * 1e4)
    ax.set_yticklabels(['{}e4'.format(int(tick) / int(1e4)) for tick in ax.get_yticks()])
    ax.set_xticks(n_structures)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

fig.tight_layout()
plt.show()


