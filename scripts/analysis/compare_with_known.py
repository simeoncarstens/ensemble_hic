import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from csb.bio.utils import rmsd
from csb.bio.io.wwpdb import StructureParser
from scipy.cluster.vq import kmeans2

from ensemble_hic.setup_functions import parse_config_file
from ensemble_hic.analysis_functions import load_sr_samples

if True:
    config_file = sys.argv[1]
else:
    pass
data_dir = os.path.expanduser('~/projects/ensemble_hic/data/proteins/')
config = parse_config_file(config_file)

n_structures = int(config['general']['n_structures'])
n_replicas = 80
n_samples = int(config['general']['n_samples']) + 1
burnin = 10000
save_figures = False

data_file = config['general']['data_file']
fnames = [data_file[:4], data_file[5:9]]

knowns = [StructureParser(data_dir + fname1 + '.pdb').parse().get_coordinates(['CA'])]
if fname2 == 'none':
    knowns.append(StructureParser(data_dir + fname2 + '.pdb').parse().get_coordinates(['CA']))
knowns = np.array(knowns) / 3.8

output_folder = config['general']['output_folder']
samples = load_sr_samples(output_folder + 'samples/', n_replicas, n_samples, 
                          config['replica']['dump_interval'],
                          burnin=burnin)
ens = np.array([sample.variables['structures'].reshape(-1, len(known1), 3)
                for sample in  samples])
ens_flat = ens.reshape(ens.shape[0] * ens.shape[1], -1, 3)

figures_folder = output_folder + 'analysis/compare_to_known/'
if not os.path.exists(output_folder + 'analysis'):
    os.makedirs(directory)
if not os.path.exists(figures_folder):
    os.makedirs(directory)

if True:
    ## plot histograms of RMSDs to known structures
    rmsds = [map(lambda x: rmsd(known, x), ens_flat) for known in knowns]
    max_rmsd = np.max(rmsds)
    min_rmsd = np.min(rmsds)
    

    fig = plt.figure()
    for i, known in enumerate(knowns):
        ax = fig.add_subplot(len(knowns),1,1)
        ax.hist(rmsds1, bins=len(ens_flat))
        ax.set_xlabel('RMSD to ' + fnames[i])
        ax.set_xlim((min_rmsd, max_rmsd))
    
    fig.tight_layout()
    if save_figures:
        plt.savefig(figures_folder + 'compare_to_known/RMSDs.pdf')
    else:
        plt.show()
