import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_samples_from_cfg

sims = ('/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_it2_fixed_40structures_sn_130replicas/',
        '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_fixed_40structures_sn_140replicas/',
        
        )

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

fig = plt.figure()
for i, sim in enumerate(sims, 1):
    config_file = sim + 'config.cfg'
    settings = parse_config_file(config_file)
    samples = load_samples_from_cfg(config_file)
    p = make_posterior(settings)
    fwm = p.likelihoods['ensemble_contacts'].forward_model
    energies = np.array(map(lambda x: -p.log_prob(**x.variables), samples))
    map_sample = samples[np.argmin(energies)]

    ax = fig.add_subplot(1, len(sims), i, aspect='equal')
    ax.scatter(fwm.data_points[:,2], fwm(**map_sample.variables), s=8,
               c=('blue', 'orange')[i-1])
    maks = np.concatenate((fwm.data_points[:,2], fwm(**map_sample.variables))).max()
    ax.plot((0, maks * 1.1), (0, maks * 1.1), ls='--', c='r')
    print np.corrcoef(fwm.data_points[:,2], fwm(**map_sample.variables))[0,1]
    ax.set_xlim(0, maks * 1.1)
    ax.set_ylim(0, maks * 1.1)
    ax.set_xlabel('experimental counts')
    ax.set_ylabel('back-calculated counts')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.show()
    
