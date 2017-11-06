import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from csb.bio.utils import rmsd
from csb.bio.io.wwpdb import StructureParser
from scipy.cluster.vq import kmeans2

from ensemble_hic.setup_functions import parse_config_file
from ensemble_hic.analysis_functions import load_sr_samples

if not True:
    config_file = sys.argv[1]
else:
    # config_file = '/scratch/scarste/ensemble_hic/proteins/1pga_1shf_fwm_poisson_10structures_sn_80replicas/config.cfg'
    pass
data_dir = os.path.expanduser('~/projects/ensemble_hic/data/proteins/')
config = parse_config_file(config_file)

n_structures = int(config['general']['n_structures'])
n_replicas = 80
n_samples = int(config['replica']['n_samples']) + 1
burnin = 30000
save_figures = True

data_file = config['general']['data_file']
data_file = data_file[-data_file[::-1].find('/'):]
fnames = [data_file[:4], data_file[5:9]]

knowns = [StructureParser(data_dir + fnames[0] + '.pdb').parse().get_coordinates(['CA'])]
if fnames[1] != 'none':
    knowns.append(StructureParser(data_dir + fnames[1] + '.pdb').parse().get_coordinates(['CA']))
knowns = np.array(knowns) / 3.8

output_folder = config['general']['output_folder']
samples = load_sr_samples(output_folder + 'samples/', n_replicas, n_samples, 
                          int(config['replica']['samples_dump_interval']),
                          burnin=burnin)
ens = np.array([sample.variables['structures'].reshape(-1, len(knowns[0]), 3)
                for sample in  samples])
ens_flat = ens.reshape(ens.shape[0] * ens.shape[1], -1, 3)

figures_folder = output_folder + 'analysis/compare_to_known/'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

if True:
    ## plot histograms of RMSDs to known structures
    rmsds = [map(lambda x: rmsd(known, x), ens_flat) for known in knowns]
    max_rmsd = np.max(rmsds)
    min_rmsd = np.min(rmsds)
    

    fig = plt.figure()
    for i, known in enumerate(knowns):
        ax = fig.add_subplot(len(knowns),1,i+1)
        ax.hist(rmsds[i], bins=np.linspace(0.0,6,np.sqrt(len(ens_flat))))
        ax.set_xlabel('RMSD to ' + fnames[i])
        #ax.set_xlim((0.3, 4.0))
    
    fig.tight_layout()
    if save_figures:
        plt.savefig(figures_folder + 'RMSDs.pdf')
    else:
        plt.show()
