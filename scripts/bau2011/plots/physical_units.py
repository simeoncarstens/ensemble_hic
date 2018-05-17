import os, sys
import numpy as np
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/'))
from process_5C_data import make_data, ignored_rfs, no_data_rfs, make_beads_RF_list

density = 12e6  # bp/micron^3

dfile = os.path.expanduser('~/projects/ensemble_hic/data/bau2011/K562.txt')
_, data_matrix = make_data(dfile, ignored_rfs, no_data_rfs)

fw_beads_RFs, rv_beads_RFs = make_beads_RF_list(data_matrix)
beads_RFs = np.vstack((fw_beads_RFs, rv_beads_RFs))
sorted_beads_RFs = beads_RFs[beads_RFs[:,0].argsort()]

RF_lens = (sorted_beads_RFs[:,2] - sorted_beads_RFs[:,1])
RF_vols = RF_lens / density # in micron^3
RF_radii = (RF_vols * 3. / 4 / np.pi) ** (1./3.) * 10 ** 3 # in nm
bead_radii = np.vstack((sorted_beads_RFs[:,0], RF_radii)).T
br = np.loadtxt(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/bead_radii_fixed.txt'))
scale_factor = bead_radii[0,1] / br[0]
