import os
import numpy

from ensemble_hic.analysis_functions import load_sr_samples
from ensemble_hic.setup_functions import parse_config_file, make_posterior

burnin = 33000

what = 'hairpin_s'
#what = 'proteins'

if what == 'hairpin_s':
    sd = ((1,  28),
          (2,  30),
          (3,  36),
          (4,  40),
          (5,  45),
          (10, 58))

    base = '/scratch/scarste/ensemble_hic/hairpin_s/hairpin_s_fwm_poisson_new_'
    base += 'it3_{}structures_sn_{}replicas/'

if what == 'proteins':
    sd = ((1, 58),
          (2, 54),
          (3, 66),
          (4, 70),
          (5, 74),
          (10, 98))

    base = '/scratch/scarste/ensemble_hic/proteins/1pga_1shf_fwm_poisson_new_'
    base += 'it3_{}structures_sn_{}replicas/'

mean_gammas = []
for n_structures, n_replicas in sd:
    settings = parse_config_file(base.format(n_structures, n_replicas) + 'config.cfg')
    samples_folder = settings['general']['output_folder'] + 'samples/'
    n_replicas = int(settings['replica']['n_replicas'])
    n_samples = int(settings['replica']['n_samples'])
    dump_interval = int(settings['replica']['samples_dump_interval'])

    samples = load_sr_samples(samples_folder, n_replicas, n_samples,
                              dump_interval, burnin)
    gammas = np.array([sample.variables['norm'] for sample in samples])
    mean_gammas.append(gammas.mean())

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

fig, ax = plt.subplots()
ax.plot(np.array(sd)[:,0], mean_gammas, ls='--', marker='o')
ax.set_xticks(np.array(sd)[:,0])
ax.set_xlabel('# of states')
ax.set_ylabel(r'$<\gamma>$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.show()
