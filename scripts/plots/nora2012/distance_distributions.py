import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from ensemble_hic.analysis_functions import load_sr_samples

probes = (
    ('pEN1',  100423573, 100433412, 'Linx'),
    ('pEN2',  100622909, 100632521, 'Xite'),
    ('pLG1',  100456274, 100465704, 'Linx'),	
    ('pLG10', 100641750, 100646253, 'Dxpas34'),
    ('pLG11', 100583328, 100588266, 'Chic1'),
    ('X3',    100512892, 100528952, 'Cdx4'),
    ('X4',    100557118, 100569724, 'Chic1')
    )

dpath = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/giorgetti2014/DNA_FISH_resume.xlsx')
from xlrd import open_workbook
wb = open_workbook(dpath)
sheet = wb.sheets()[0]
table = np.array([np.array(sheet.row_values(j))[1:13]
                  for j in [2,3]+range(7, sheet.nrows)])
data = {'{}:{}'.format(x[0], x[1]): np.array([float(y) for y in x[2:] if len(y) > 0])
        for x in table.T}

bead_size = 3000
#bead_size = 15000
region_start = 100378306

simlist = ((1, 298, 50001, 30000,  '_it3', '', 1000),
           (5, 218, 50001, 30000,  '_it3', '', 1000),
           (10, 297, 50001, 30000, '', '', 500),
           (20, 309, 50001, 30000, '_it3', '', 1000),
           (20, 309, 50001, 30000, '_it3', '_rep3', 1000),
           (20, 309, 50001, 30000, '_it3', '_rep4', 1000),
           (30, 330, 32001, 20000, '', '_rep1', 1000),
           (30, 330, 43001, 30000, '', '_rep2', 1000),
           (30, 330, 43001, 30000, '', '_rep3', 1000),
           (40, 330, 33001, 20000, '_it2', '', 1000),
           (40, 330, 33001, 20000, '_it2', '_rep1', 1000),
           (40, 330, 33001, 20000, '_it2', '_rep2', 1000),
           # (20, 330, 25001, 15000, '', '_noii3', 1000),
           # (40, 330, 16001, 8000, '', '_noii3', 1000)
           )

n_structures, n_replicas, n_samples, burnin, it, rep, di = simlist[-9]
#n_structures, n_replicas, n_samples, burnin, it, rep, di = simlist[int(sys.argv[1])]
n_beads = 308
#n_beads = 62

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains{}{}_{}structures_{}replicas/'.format(it, rep, n_structures, n_replicas)

# sim_path = '/scratch/scarste/ensemble_hic/nora2012/15kbbins_bothdomains_it2_40structures_111replicas/'
# n_replicas = 39
# di = 1000
# n_samples = 133000
# burnin = 70000

samples = load_sr_samples(sim_path + 'samples/', n_replicas,
                          n_samples, di, burnin)
X = np.array([s.variables['structures'].reshape(n_structures,-1,3) for s in samples])

Xflat = X.reshape(-1, n_beads, 3) * 53
# Xflat = X.reshape(-1, n_beads, 3) * (5 * 53 ** 3) ** 0.333

if True:
    Xflats_alber = []
    for i in (100, 1000, 10000):
        Xflat_temp = np.load('/scratch/scarste/ensemble_hic/alber/nora2012/clipped_nosphere_n{}/ensemble.npy'.format(i))
        seq_ds = np.array([np.linalg.norm(Xflat_temp[:,i+1] - Xflat_temp[:,i], axis=1)
                           for i in range(307)])
        Xflats_alber.append(Xflat_temp * 53.0 / seq_ds.mean())
    
get_bead = lambda p: int((np.mean(p[1:3]) - region_start) / bead_size)


n_bins = int(np.sqrt(len(Xflat)) / 3)
# axes_flat[0].set_visible(False)
# combs = ((2,1), (2,4), (0,4), (6,1), (5,1), (5,6), (0,3))
combs = ((1,2), (1,6), (1,5), (5,6), (2,1), (0,3), (1,4)) 
mapping = (data['pEN2:pLG1'], data['pEN2:X4'], data['pEN2:X3'], data['X4:X3'],
           data['pLG1:pEN2'], data['Dxpas34:pEN1'], data['pEN2:pLG11'])

def plot_distance_hists(ax, i, l1, l2):
    ax.hist(np.linalg.norm(Xflat[:,get_bead(probes[l1])] -
                           Xflat[:,get_bead(probes[l2])],
                           axis=1),
            bins=n_bins, histtype='step', label='model',
            normed=True, color='black', lw=1)
    ax.hist(mapping[i-1],
            bins=int(np.sqrt(len(mapping[i-1]))), histtype='step',
            label='FISH', normed=True, color='gray', lw=1)
    #ax.axvline(1 * 53, color='g', ls='--', label='bead diameter')
    ax.text(0.5, 0.8, '{} - {}'.format(probes[l1][0], probes[l2][0]),
            transform=ax.transAxes)
    ax.set_yticks(())
    ax.set_xticks((0, 400, 800))
    ax.set_xlim((0, 1200))#900))
    for x in ('left', 'top', 'right'):
        ax.spines[x].set_visible(False)

def plot_alber_distance_hists(ax, i, l1, l2):

    from ensemble_hic.analysis_functions import calculate_KL_KDE_log
    from scipy.linalg import norm
    h = lambda p, q: norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    ds = np.linalg.norm(Xflat[:,get_bead(probes[l1])] -
                        Xflat[:,get_bead(probes[l2])],
                        axis=1),
    for j in range(len(Xflats_alber)):
        alber_ds = np.linalg.norm(Xflats_alber[j][:,get_bead(probes[l1])] -
                                  Xflats_alber[j][:,get_bead(probes[l2])],
                                  axis=1)
        ax.hist(alber_ds,
                bins=n_bins / 3, histtype='step',
                normed=True, color=('blue', 'red', 'green')[j], lw=1)
        # print calculate_KL_KDE_log((alber_ds, mapping[i-1])), '(n={})'.format(len(Xflats_alber[j]) / 2)
        #print h(alber_ds, mapping[i-1]), '(n={})'.format(len(Xflats_alber[j]) / 2)
    # ax.hist(mapping[i-1],
    #         bins=int(np.sqrt(len(mapping[i-1]))), histtype='step',
    #         label='FISH', normed=True, color='gray', lw=1)
    #ax.axvline(1 * 53, color='g', ls='--', label='bead diameter')
    # ax.text(0.5, 0.8, '{} - {}'.format(probes[l1][0], probes[l2][0]),
    #         transform=ax.transAxes)
    ax.set_yticks(())
    ax.set_xticks((0, 400, 800))
    ax.set_xlim((0, 1200))#900))
    for x in ('left', 'top', 'right'):
        ax.spines[x].set_visible(False)


def plot_all_hists(axes):

    for i, (l1, l2) in enumerate(combs):
        plot_distance_hists(axes[i], i, l1, l2)

def plot_all_hists_alber(axes):

    for i, (l1, l2) in enumerate(combs):
        plot_alber_distance_hists(axes[i], i, l1, l2)

if False:
    fig, axes = plt.subplots(6, 3, sharey=True)
    for i in range(3):
        pairs = [(axes[2*i,j], axes[2*i+1,j]) for j in range(3)]
        for ax1, ax2 in pairs:
            ax1.get_shared_x_axes().join(ax1, ax2)
            ax1.set_xticklabels([])
    # for ax in axes.ravel()[-2:]:
    #     ax.set_visible(False)
    # for ax in axes.ravel()[:-2]:
    #     ax.set_ylabel('count')
    #     ax.set_xlabel('distance [nm]')
        
    plot_all_hists_alber(axes[1::2].ravel())

if False:
    fig, axes = plt.subplots(3, 3, sharey=True)
    for ax in axes.ravel()[-2:]:
        ax.set_visible(False)
    for ax in axes.ravel()[:-2]:
        ax.set_ylabel('count')
        ax.set_xlabel('distance [nm]')
        
    plot_all_hists(np.array(axes).ravel())
    path = os.path.expanduser('~/projects/ehic-paper/nmeth/supplementary_information/figures/nora_distance_histograms/')
    fig.savefig(path + '{}structures{}.svg'.format(n_structures, rep))
    fig.savefig(path + '{}structures{}.pdf'.format(n_structures, rep))

           
# for i, (l1, l2) in enumerate(combs):
#     cax = axes_flat[i]
#     cax.hist(np.linalg.norm(Xflat[:,get_bead(probes[l1])] -
#                             Xflat[:,get_bead(probes[l2])],
#                             axis=1),
#              bins=n_bins, histtype='step', label='model',
#              normed=True)
#     cax.hist(mapping[i-1],
#              bins=int(np.sqrt(len(mapping[i-1]))), histtype='step',
#              label='FISH', normed=True)
#     cax.axvline(1 * 53, color='g', ls='--', label='bead diameter')
#     cax.text(0.5, 0.8, '{} - {}'.format(probes[l1][0], probes[l2][0]),
#              transform=cax.transAxes)
#     cax.set_yticks(())
#     cax.set_xticks((0, 250, 500, 750, 1000))
#     for x in ('left', 'top', 'right'):
#         cax.spines[x].set_visible(False)
# axes_flat[-1].set_visible(False)
# axes_flat[-2].set_visible(False)
# axes[2,1].set_xlabel('model distance [nm]')
# l = axes_flat[0].legend(frameon=False)
# l.set_visible(False)
# handles, labels = axes_flat[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc=(0.4, 0.15), frameon=False)
# # a = axes_flat[-1].annotate('model distance [nm]', xy=(0.5,0.5), xytext=(0,0),
# #                            arrowprops=dict(arrowstyle="->"),
# #                            )#transform=axes_flat[-1].transAxes)

# plt.rc('font', size=8)
# fig.tight_layout()
# plt.show()
