import numpy as np
import sys
import os

from csb.bio.utils import rmsd, fit_transform
from csb.bio.io.wwpdb import StructureParser

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg
from ensemble_hic.analysis_functions import write_ensemble
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/misc/'))
from simlist import simulations

sims = simulations['1pga_1shf_fwm_poisson_new_fixed_it3']
cpath = sims['common_path']
n_structures = sims['n_structures']
odirs = sims['output_dirs']

MAP_samples = []
for (n, odir) in zip(n_structures, odirs):
    cfg_file = cpath + odir + '/config.cfg'
    p = make_posterior(parse_config_file(cfg_file))
    samples = load_samples_from_cfg(cfg_file, burnin=60000)
    Es = map(lambda x: -p.log_prob(**x.variables), samples)
    MAP_sample = samples[np.argmin(Es)]
    MAP_samples.append(MAP_sample.variables['structures'].reshape(n, 56, 3))
    
MAP_samples_flat = np.array([x for y in MAP_samples for x in y])

invar_rmsd = lambda x, y: min(rmsd(x, y), rmsd(x, -y))
tmp = StructureParser(os.path.expanduser('~/projects/ensemble_hic/data/proteins/1pga.pdb'))
ref_1pga = tmp.parse().get_coordinates(['CA'])
tmp = StructureParser(os.path.expanduser('~/projects/ensemble_hic/data/proteins/1shf.pdb'))
ref_1shf = tmp.parse().get_coordinates(['CA'])
rmsds_to_1pga = map(lambda x: invar_rmsd(ref_1pga, x), MAP_samples_flat)
rmsds_to_1shf = map(lambda x: invar_rmsd(ref_1shf, x), MAP_samples_flat)

labels = (np.array(rmsds_to_p1) < np.array(rmsds_to_p2))
for i, s in enumerate(MAP_samples_flat):
    ref = (ref_1pga, ref_1shf)[labels[i]]
    if rmsd(ref, -s) < rmsd(ref, s):
        MAP_samples_flat[i] *= -1
    MAP_samples_flat[i] = fit_transform(ref, MAP_samples_flat[i])

write_ensemble(MAP_samples_flat, '/tmp/protein_MAPs.pdb')
