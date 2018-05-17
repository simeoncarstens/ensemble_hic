import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from csb.bio.utils import radius_of_gyration

from ensemble_hic.setup_functions import parse_config_file
from ensemble_hic.analysis_functions import load_samples_from_cfg

import sys
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/bau2011/plots/'))
from physical_units import scale_factor
from lib import br
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/misc/'))
from simlist import simulations

which = ('K562_new_smallercd_nosphere_fixed',
         'GM12878_new_smallercd_nosphere_fixed'
         )

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

fig = plt.figure()
ax = fig.add_subplot(111)
for j, sim in enumerate(which):
    current = simulations[sim]
    all_rogs = []
    n_structures = current['n_structures']
    for n in current['n_structures']:
        cp = current['common_path']
        path = cp + filter(lambda x: '{}stru'.format(n) in x,
                           current['output_dirs'])[0]
        print path
        config_file = path + '/config.cfg'
        samples = load_samples_from_cfg(config_file)
        X = np.vstack([x.variables['structures'].reshape(n, -1, 3) for x in samples])
        rogs = map(lambda y: radius_of_gyration(y, br**3), X)
        all_rogs.append(np.array(rogs) * scale_factor)
    
    # ax.plot(n_structures, [np.mean(all_rogs[i]) for i in range(len(n_structures))],
    #         label=sim, marker='o')
    ax.errorbar(n_structures, [all_rogs[i].mean() for i in range(len(n_structures))],
                [all_rogs[i].std() for i in range(len(n_structures))],
                label=('K562', 'GM12878')[j], marker='o')
ax.set_xlabel('# of states')
ax.set_ylabel(r'$<r_{gyr}>$ [nm]')
ax.set_xticks((1, 7, 10, 15, 20, 30, 40, 50))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()
