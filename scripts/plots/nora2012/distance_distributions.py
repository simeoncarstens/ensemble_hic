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

## combinations of probes tested
combinations = ((1,2), (1,6), (1,5), (5,6), (2,1), (0,3), (1,4)) 
    

def calculate_data(config_file, fish_data_file):
    

    from xlrd import open_workbook
    from ensemble_hic.setup_functions import parse_config_file
    
    wb = open_workbook(fish_data_file)
    sheet = wb.sheets()[0]
    table = np.array([np.array(sheet.row_values(j))[1:13]
                      for j in [2,3]+range(7, sheet.nrows)])
    data = {'{}:{}'.format(x[0], x[1]): np.array([float(y) for y in x[2:] if len(y) > 0])
            for x in table.T}

    bead_size = 3000
    region_start = 100378306
    n_beads = 308

    settings = parse_config_file(config_file)
    output_folder = settings['general']['output_folder']
    n_structures = int(settings['general']['n_structures'])
    samples = load_sr_samples(output_folder + 'samples/',
                              int(settings['replica']['n_replicas']),
                              50001, 1000, 30000)
    X = np.array([s.variables['structures'].reshape(n_structures,-1,3) for s in samples])
    Xflat = X.reshape(-1, n_beads, 3) * 53
    
    get_bead = lambda p: int((np.mean(p[1:3]) - region_start) / bead_size)
    
    mapping = (data['pEN2:pLG1'], data['pEN2:X4'], data['pEN2:X3'], data['X4:X3'],
               data['pLG1:pEN2'], data['Dxpas34:pEN1'], data['pEN2:pLG11'])

    isd_distance_dists = [np.linalg.norm(Xflat[:,get_bead(probes[l1])] -
                                         Xflat[:,get_bead(probes[l2])],
                                         axis=1)
                          for (l1, l2) in combinations]
    fish_distance_hists = [mapping[i-1] for i in range(len(combinations))]

    return isd_distance_dists, fish_distance_hists

def plot_distance_hists(ax, isd_distances, fish_distances, l1, l2):

    n_bins = int(np.sqrt(len(isd_distances)) / 3)
    ax.hist(isd_distances, bins=n_bins, histtype='step', label='model',
            normed=True, color='black', lw=1)
    ax.hist(fish_distances,
            bins=int(np.sqrt(len(fish_distances))), histtype='step',
            label='FISH', normed=True, color='gray', lw=1)
    ax.text(0.5, 0.8, '{} - {}'.format(probes[l1][0], probes[l2][0]),
            transform=ax.transAxes)
    ax.set_yticks(())
    ax.set_xticks((0, 400, 800))
    ax.set_xlim((0, 1200))
    for x in ('left', 'top', 'right'):
        ax.spines[x].set_visible(False)

        
def plot_all_hists(axes, plot_data_file):

    isd_distances, fish_distances = np.load(plot_data_file)
    
    for i, (l1, l2) in enumerate(combinations):
        plot_distance_hists(axes[i], isd_distances[i], fish_distances[i], l1, l2)


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


if __name__ == "__main__":

    import sys
    from cPickle import dump

    config_file = sys.argv[1]
    fish_data_file = sys.argv[2]
    output_file = sys.argv[3]
    isd_distance_dists, fish_distance_hists = calculate_data(config_file, fish_data_file)

    with open(output_file, "w") as opf:
        dump((isd_distance_dists, fish_distance_hists), opf)
