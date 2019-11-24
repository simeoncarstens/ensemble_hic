import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from csb.bio.utils import rmsd, radius_of_gyration as rog, distance_matrix
from ensemble_hic.analysis_functions import load_sr_samples


fwm = lambda x: 0.5 * (x / np.sqrt(1 + x * x) + 1)
cgen_ss = lambda d, a, cutoff, offset: np.triu((fwm(a * (cutoff - d))), offset)
random = lambda n: np.random.choice(np.arange(len(X)), n)


def plot_binding_sites(ax, filenames):
    '''
    ChIP-seq peaks from https://www.encodeproject.org/experiments/ENCSR362VNF/
    Filenames is some iterable containing the paths to both replicates
    (ENCFF001ZQW.broadPeak.bed, ENCFF001ZQX.broadPeak.bed)
    '''

    # start of Tsix TAD
    coords_min = 100378306
    # end of Xist TAD
    coords_max = 101298738

    bead_size = (coords_max - coords_min) / 308.0

    def load_and_filter(filename):
        encode_peaks_bed = np.loadtxt(filename, dtype=str)

        # filter for genomic region of itnerest
        encode_peaks = np.array([(int(x[1]), int(x[2])) 
                                 for x in encode_peaks_bed if x[0] == 'chrX' 
                                 and int(x[1]) >= coords_min 
                                 and int(x[2]) <= coords_max
                                 ])
        return encode_peaks

    encode_peaks = np.concatenate([load_and_filter(filename) for filename in filenames])
    encode_peaks_beads = (encode_peaks - coords_min) / bead_size

    for i, peak in enumerate(encode_peaks_beads):
        ax.plot((peak, peak), (0, 2.5e-5), color="black", linewidth=0.5, 
                label="CTCF binding sites" if i == 0 else None)


    
def calculate_TAD_boundaries(X):

    np.random.seed(32)
    
    cutoff = 2.0
    a = 10.0
    offset = 6

    if False:
        scores_pop = np.array(map(lambda x: find_TADs_pop_py(x, a, cutoff, offset),
                                  X / 53.0))
        scores = np.array(map(lambda x: find_TADs_py(x, a, cutoff, offset),
                              X.reshape(-1,308,3) / 53.0))

    else:
        # Cython implementation, only a little faster unfortunately
        from ensemble_hic.TAD_analysis_PNAS_c import find_TADs, find_TADs_pop

        scores_pop = find_TADs_pop(X / 53.0, a, cutoff, offset)
        scores = find_TADs(X.reshape(-1, 308, 3) / 53.0, a, cutoff, offset)

    return scores_pop, scores
    

def find_TADs_py(x, a, cutoff, offset):
    d = distance_matrix(x)
    c = cgen_ss(d, a, cutoff, offset)
    j = np.arange(len(x))
    counts = np.array([c[:i,:i].sum() + c[i:,i:].sum() for i in j])
    areas  = j**2 + (len(x) - j)**2

    return np.argmax(counts.astype('d') / areas)

def find_TADs_pop_py(X, a, cutoff, offset):
    d = np.array([squareform(pdist(x)) for x in X])
    c = np.sum(map(lambda sd: cgen_ss(sd, a, cutoff, offset), d), axis=0) / len(d)
    j = np.arange(len(X[0]))
    counts = np.array([c[:i,:i].sum() + c[i:,i:].sum() for i in j])
    areas  = j**2 + (len(X[0]) - j)**2

    return np.argmax(counts.astype('d') / areas)

def plot_TAD_boundary_hists(ax, data_file, peak_filenames):

    from collections import OrderedDict
    
    scores_pop, scores = np.load(data_file, allow_pickle=True)

    hargs = dict(alpha=0.5, histtype='stepfilled', normed=True)
    ax.hist(scores, label='single structures', color='gray',
                 bins=np.arange(0,308,2), **hargs)
    ax.hist(scores_pop, label='structure populations', color='black',
            bins=np.arange(0,308,1), **hargs)
    plot_binding_sites(ax, peak_filenames)
    print scores.mean(), scores_pop.mean()
    print scores.min(), scores.max()
    ax.set_xlabel('TAD boundary position [beads]')
    #ax.set_xlim(50, 225)
    ax.yaxis.set_visible(False)
    for spine in ('top', 'left', 'right'):
        ax.spines[spine].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False)
    ax.set_yscale('log')



if __name__ == "__main__":

    import sys
    from cPickle import dump
    from ensemble_hic.setup_functions import parse_config_file

    cfg_file = sys.argv[1]
    output_file = sys.argv[2]

    settings = parse_config_file(cfg_file)
    n_replicas = int(settings['replica']['n_replicas'])
    n_structures = int(settings['general']['n_structures'])

    scale_factor = 53
    
    samples = load_sr_samples(settings['general']['output_folder'] + 'samples/',
                              n_replicas, 50001, 1000, 30000)
    X = np.array([x.variables['structures'].reshape(n_structures, 308, 3)
                  for x in samples]) * scale_factor

    boundaries = calculate_TAD_boundaries(X)

    with open(output_file, "w") as opf:
        dump(boundaries, opf)
