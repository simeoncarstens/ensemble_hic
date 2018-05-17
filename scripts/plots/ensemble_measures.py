import sys, os
import numpy as np
import matplotlib.pyplot as plt

from csb.bio.utils import radius_of_gyration

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg

if not True:
    config_file1 = sys.argv[1]
    config_file2 = sys.argv[2]
else:
    config_file2 = '/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_40structures_sn_109replicas/config.cfg'
    config_file1 = '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_30structures_sn_122replicas/config.cfg'

labels = ['GM12878, n=30', 'K562, n=40']

fig = plt.figure()
ax = fig.add_subplot(111)
for i, config_file in enumerate((config_file1, config_file2)):
    settings = parse_config_file(config_file)
    n_beads = int(settings['general']['n_beads'])
    n_structures = int(settings['general']['n_structures'])
    samples = load_samples_from_cfg(config_file)
    X = np.array([s.variables['structures'].reshape(n_structures, -1, 3)
                  for s in samples])
    rgs = np.array(map(radius_of_gyration, X.reshape(-1, n_beads, 3)))
    ax.hist(rgs, bins=int(np.sqrt(len(rgs))), alpha=0.6, label=labels[i])
ax.set_xlabel('radius of gyration')
ax.set_yticks([])
ax.legend()
plt.show()

