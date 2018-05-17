import sys
import numpy as np
import matplotlib.pyplot as plt

from ensemble_hic.setup_functions import make_posterior, parse_config_file

#sys.argv[1] = '/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017_it2_20structures_sn_366replicas/config.cfg'
# sys.argv[1] = '/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017_it2_10structures_sn_438replicas/config.cfg'
sys.argv[1] = '/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017_it5_50structures_sn_629replicas/config.cfg'
sys.argv[1] = '/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017_it3_100structures_sn_725replicas/config.cfg'

config_file = sys.argv[1]
settings = parse_config_file(config_file)

output_folder = settings['general']['output_folder']
n_replicas = int(settings['replica']['n_replicas'])

p = make_posterior(settings)

n_beads = p.priors['nonbonded_prior'].forcefield.n_beads
L = p.likelihoods['ensemble_contacts']
FWM = L.forward_model
d = FWM.data_points

dump_interval = int(settings['replica']['samples_dump_interval'])

for i in range(0, 200000, dump_interval)[::-1]:
    try:
        path = output_folder
        path += 'samples/samples_replica{}_{}-{}.pickle'.format(n_replicas,
                                                                i, i+dump_interval)
        s = np.load(path)
        print path
        break
    except Exception, msg:
        pass

md = FWM(**s[-1].variables)

sel = np.where((d[:,1] - d[:,0]) >0)
md_scatter = md[sel]
d_scatter = d[sel]

if True:
    sorted_inds = np.argsort(d_scatter[:,0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d_scatter[sorted_inds,0],
            md_scatter[sorted_inds] - d_scatter[sorted_inds,2])
    mr = p.priors['backbone_prior']._mol_ranges
    [ax.plot((x,x),(0,300), ls='--', c='r') for x in mr]
    plt.show()

m = np.zeros((n_beads, n_beads))
m[d[:,0], d[:,1]] = md
m[d[:,1], d[:,0]] = d[:,2]

fig = plt.figure()
ax1 = fig.add_subplot(121)
ms1 = ax1.matshow(np.log(m+1))
ax1.set_xticks(())
ax1.set_yticks(())

ax2 = fig.add_subplot(122)
ms1 = ax2.scatter(d_scatter[:,2], md_scatter,
                  c=d_scatter[:,1]-d_scatter[:,0], vmin=0, vmax=10, s=1)
ax2.set_xlabel('data counts')
ax2.set_ylabel('mock data counts')
max_count = d_scatter[:,2].max()
ax2.set_xlim((0, max_count))
ax2.set_ylim((0, max_count))
ax2.set_aspect('equal')
ax2.plot((0, max_count), (0, max_count), ls='--', c='r')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(ms1, cax=cax)
cb.set_label('linear distance')

fig.tight_layout()
plt.show()


