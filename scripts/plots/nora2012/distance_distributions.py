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
region_start = 100378306

simlist = ((1, 298, 50001, 30000,  '_it3', '', 1000),
           (5, 218, 50001, 30000,  '_it3', '', 1000),
           (10, 297, 50001, 30000, '', '', 500),
           (20, 309, 50001, 30000, '_fixed_it3', '_rep1', 1000),
           (20, 309, 50001, 30000, '_fixed_it3', '_rep2', 1000),
           (20, 309, 50001, 30000, '_fixed_it3', '_rep3', 1000),
           (30, 330, 32001, 20000, '', '_rep1', 1000),
           (30, 330, 43001, 30000, '', '_rep2', 1000),
           (30, 330, 43001, 30000, '', '_rep3', 1000),
           (40, 330, 33001, 20000, '_it2', '', 1000),
           (40, 330, 33001, 20000, '_it2', '_rep1', 1000),
           (40, 330, 33001, 20000, '_it2', '_rep2', 1000),
           )

n_structures, n_replicas, n_samples, burnin, it, rep, di = simlist[5]
n_beads = 308

sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains{}{}_{}structures_{}replicas/'.format(it, rep, n_structures, n_replicas)

samples = load_sr_samples(sim_path + 'samples/', n_replicas,
                          n_samples, di, burnin)
X = np.array([s.variables['structures'].reshape(n_structures,-1,3) for s in samples])

Xflat = X.reshape(-1, n_beads, 3) * 53
    
get_bead = lambda p: int((np.mean(p[1:3]) - region_start) / bead_size)


n_bins = int(np.sqrt(len(Xflat)) / 3)
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
    ax.text(0.5, 0.8, '{} - {}'.format(probes[l1][0], probes[l2][0]),
            transform=ax.transAxes)
    ax.set_yticks(())
    ax.set_xticks((0, 400, 800))
    ax.set_xlim((0, 1200))
    for x in ('left', 'top', 'right'):
        ax.spines[x].set_visible(False)

        
def plot_all_hists(axes):

    for i, (l1, l2) in enumerate(combs):
        plot_distance_hists(axes[i], i, l1, l2)


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
