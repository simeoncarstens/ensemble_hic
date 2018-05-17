import os, sys, numpy as np
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/data/eser2017/'))
from yeastlib import chrom_lengths

## best model from Arbona et al., 2017
W = 30       # bead diameter in nm
C = 50       # chromatin compaction in bp/nm
P = 69       # persistence length in nm
L = 400      # microtubule length in nm
R = 1000     # nuclear radius in nm
W_rDNA = 194 # diameter of rDNA beads in nm; there are 150 of them
n_rDNA_beads = 150

rDNA_vol = n_rDNA_beads * 4. / 3. * np.pi * (W_rDNA / 2.0) ** 3
nuclear_vol = 4. / 3. * np.pi * R ** 3
n_beads = np.round(chrom_lengths[:,1] / float(W * C))
beads_vol = n_beads.sum() * 4. / 3. * np.pi * (W / 2.) ** 3
beads_vol_fraction = beads_vol / nuclear_vol
rDNA_vol_fraction = rDNA_vol / nuclear_vol 
rDNA_normal_beads_volume_ratio = rDNA_vol / beads_vol

rDNA_normal_ratio = W_rDNA / float(W)

## determine our bead radius by maintaining volume fraction of nucleus and
## ratio of rDNA volume to normal chromatin volume

our_n_beads = 1216
our_R_beads = (beads_vol_fraction / our_n_beads) ** (1./3.) * R
our_R_rDNA = our_R_beads * (W_rDNA / 2.) / (W / 2.)
our_n_rDNA_beads = n_rDNA_beads * (W_rDNA / 2.0) ** 3 / our_R_rDNA ** 3

