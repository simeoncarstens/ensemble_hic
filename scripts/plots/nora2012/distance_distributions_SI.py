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

region_start = 100378306

cpath = '/scratch/scarste/ensemble_hic/'

path = cpath + 'nora2012/bothdomains_fixed_it3_rep3_20structures_309replicas/samples/'
X_highres = np.array([x.variables['structures'].reshape(-1,308,3)
                      for x in load_sr_samples(path, 309, 50001, 1000, 30000)])
X_highres = X_highres.reshape(-1,308,3) * 53

path = cpath + 'nora2012/15kbbins_bothdomains_fixed_it4_20structures_148replicas/samples/'
X_lowres = np.array([x.variables['structures'].reshape(-1, 62, 3)
                      for x in load_sr_samples(path, 114, 90001, 1000, 74000)])
X_lowres = X_lowres.reshape(-1, 62, 3) * (5 * 53 ** 3) ** 0.33333

path = cpath + 'nora2012/bothdomains_prior_1structures_59replicas/samples/'
X_null = np.array([x.variables['structures'].reshape(-1, 308, 3)
                      for x in load_sr_samples(path, 59, 80001, 1000, 50000)])
X_null = X_null.reshape(-1, 308, 3) * 53

Xs_alber = []
# for i in (100, 1000, 10000):
#     X_temp = np.load(cpath +
#                      'alber/nora2012/clipped_nosphere_n{}/ensemble.npy'.format(i))
#     seq_ds = np.array([np.linalg.norm(X_temp[:,i+1] - X_temp[:,i], axis=1)
#                        for i in range(307)])
#     Xs_alber.append(X_temp * 53.0 / seq_ds.mean())

for i in (100, 1000, 10000):
    X_temp = np.load(cpath +
                     'alber/nora2012/clipped_nosphere_n{}_correctsize/ensemble.npy'.format(i))
    Xs_alber.append(X_temp)

get_bead = lambda p, bead_size: int((np.mean(p[1:3]) - region_start) / bead_size)

combs = ((1,2), (1,6), (1,5), (5,6), (2,1), (0,3), (1,4)) 
mapping = (data['pEN2:pLG1'], data['pEN2:X4'], data['pEN2:X3'], data['X4:X3'],
           data['pLG1:pEN2'], data['Dxpas34:pEN1'], data['pEN2:pLG11'])

def plot_distance_hists(ax, X, i, l1, l2, bead_size, ls):
    ax.hist(np.linalg.norm(X[:,get_bead(probes[l1], bead_size)] -
                           X[:,get_bead(probes[l2], bead_size)],
                           axis=1),
            bins=int(np.sqrt(len(X)) / 3.0), histtype='step',# label='model',
            normed=True, color='black', lw=2, ls=ls)

def plot_FISH_hists(ax, i, l1, l2):
    ax.hist(mapping[i-1],
            bins=int(np.sqrt(len(mapping[i-1]))), histtype='step',
            #label='FISH',
            normed=True, color='gray', lw=2)

def plot_alber_distance_hists(ax, i, l1, l2):

    from ensemble_hic.analysis_functions import calculate_KL_KDE_log
    from scipy.linalg import norm

    bead_size = 3000
    h = lambda p, q: norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    for j in range(len(Xs_alber)):
        alber_ds = np.linalg.norm(Xs_alber[j][:,get_bead(probes[l1], bead_size)] -
                                  Xs_alber[j][:,get_bead(probes[l2], bead_size)],
                                  axis=1)
        ax.hist(alber_ds,
                bins=int(np.sqrt(len(alber_ds)) / 3.0), histtype='step',
                normed=True,
                #color=('blue', 'red', 'green')[j],
                lw=2)

def plot_all_hists(axes, X, bead_size, ls):

    for i, (l1, l2) in enumerate(combs):
        plot_distance_hists(axes[i], X, i, l1, l2, bead_size, ls)

def plot_all_FISH_hists(axes):

    for i, (l1, l2) in enumerate(combs):
        plot_FISH_hists(axes[i], i, l1, l2)

def plot_all_hists_alber(axes):

    for i, (l1, l2) in enumerate(combs):
        plot_alber_distance_hists(axes[i], i, l1, l2)

fig, axes = plt.subplots(6, 3)
for i in range(3):
    pairs = [(axes[2*i,j], axes[2*i+1,j]) for j in range(3)]
    for ax1, ax2 in pairs:
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])

plot_all_hists_alber(axes[1::2].ravel())
plot_all_hists(axes[::2].ravel(), X_highres, 3000, ls='-')
plot_all_hists(axes[::2].ravel(), X_lowres, 15000, ls='--')
plot_all_hists(axes[::2].ravel(), X_null, 3000, ls=':')
plot_all_FISH_hists(axes[1::2].ravel())
plot_all_FISH_hists(axes[::2].ravel())
for i, (l1, l2) in enumerate(combs):
    ax = axes[::2].ravel()[i]
    ax.text(0.5, 0.8, '{} - {}'.format(probes[l1][0], probes[l2][0]),
            transform=ax.transAxes)

for ax in axes.ravel():
    ax.set_yticks(())
    ax.set_xticks((0, 400, 800))
    ax.set_xlim((0, 1200))
    for x in ('left', 'top', 'right'):
        ax.spines[x].set_visible(False)

for ax in axes[-2][1:]:
    ax.set_visible(False)
for ax in axes[-1][1:]:
    ax.set_visible(False)

l1 = axes[0,0].legend(labels=('ISD (high-res, $n=20$)',
                              'ISD (low-res, $n=20$)',
                              'ISD (high-res, prior only)',
                              'FISH'))
l2 = axes[1,0].legend(labels=(r'PGS ($n=2\times100$)',
                              r'PGS ($n=2\times1000$)',
                              r'PGS ($n=2\times10000$)'))
# handles1, labels1 = axes[0,0].get_legend_handles_labels()
# handles2, labels2 = axes[0,1].get_legend_handles_labels()
handles1 = l1.legendHandles
handles2 = l2.legendHandles
labels1 = l1.texts
labels2 = l2.texts
l1.set_visible(False)
l2.set_visible(False)
new_handles = [Line2D([], [], linewidth=3, ls='--' if i == 1 else '-',
                      c=h.get_edgecolor())
               for i, h in enumerate(handles1 + handles2)]
new_handles[2].set_linestyle(':')
l3 = axes[-2,1].legend(frameon=False, handles=new_handles,
                       labels=[x.get_text() for x in labels1 + labels2])
axes[-2,1].set_visible(True)
axes[-2,1].spines['bottom'].set_visible(False)
axes[-2,1].set_xticks(())
