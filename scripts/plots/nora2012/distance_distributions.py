import os
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

simlist = ((1, 298, 50001, 30000,  '_it3'),
           (5, 218, 50001, 30000,  '_it3'),
           (10, 297, 50001, 30000, ''),
           (20, 309, 50001, 30000, '_it3'),
           (40, 330, 33001, 20000, '_it2'))

# simlist = ((1, 99, 50001, 30000,  ''),
#            (5, 99, 50001, 30000,  ''),
#            (10, 99, 50001, 30000, ''),
#            (20, 307, 50001, 30000, ''),
#            (40, 99, 49001, 30000, ''),
#            (60, 99, 32001, 18000, ''),
#            (80, 99, 24001, 12000, ''),
#            (100, 99, 18001, 8000, ''))

n_structures, n_replicas, n_samples, burnin, suffix = simlist[3]
n_beads = 308
sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains{}_{}structures_{}replicas/'.format(suffix, n_structures, n_replicas)
# sim_path = '/scratch/scarste/ensemble_hic/nora2012/bothdomains_lambdatempering_fromstates_{}structures_{}replicas/'.format(n_structures, n_replicas)
samples = load_sr_samples(sim_path + 'samples/', n_replicas,
                          n_samples, 1000, burnin)
X = np.array([s.variables['structures'].reshape(n_structures,-1,3) for s in samples])
Xflat = X.reshape(-1, n_beads, 3) * 53

get_bead = lambda p: int((np.mean(p[1:3]) - region_start) / bead_size)

fig, axes = plt.subplots(3, 3, sharey=True)

n_bins = int(sqrt(len(Xflat)) / 3)
axes_flat = axes.ravel()
# axes_flat[0].set_visible(False)
# combs = ((2,1), (2,4), (0,4), (6,1), (5,1), (5,6), (0,3))
combs = ((1,2), (1,6), (1,5), (5,6), (2,1), (0,3), (1,4)) 
mapping = (data['pEN2:pLG1'], data['pEN2:X4'], data['pEN2:X3'], data['X4:X3'],
           data['pLG1:pEN2'], data['Dxpas34:pEN1'], data['pEN2:pLG11'])
           
for i, (l1, l2) in enumerate(combs):
    cax = axes_flat[i]
    cax.hist(np.linalg.norm(Xflat[:,get_bead(probes[l1])] -
                            Xflat[:,get_bead(probes[l2])],
                            axis=1),
             bins=n_bins, histtype='step', label='model',
             normed=True)
    cax.hist(mapping[i-1],
             bins=int(np.sqrt(len(mapping[i-1]))), histtype='step',
             label='FISH', normed=True)
    cax.axvline(1 * 53, color='g', ls='--', label='bead diameter')
    cax.text(0.5, 0.8, '{} - {}'.format(probes[l1][0], probes[l2][0]),
             transform=cax.transAxes)
    cax.set_yticks(())
    cax.set_xticks((0, 250, 500, 750, 1000))
    for x in ('left', 'top', 'right'):
        cax.spines[x].set_visible(False)
axes_flat[-1].set_visible(False)
axes_flat[-2].set_visible(False)
axes[2,1].set_xlabel('model distance [nm]')
l = axes_flat[0].legend(frameon=False)
l.set_visible(False)
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.4, 0.15), frameon=False)
# a = axes_flat[-1].annotate('model distance [nm]', xy=(0.5,0.5), xytext=(0,0),
#                            arrowprops=dict(arrowstyle="->"),
#                            )#transform=axes_flat[-1].transAxes)

plt.rc('font', size=8)
fig.tight_layout()
plt.show()
