import numpy as np
import matplotlib.pyplot as plt

from ensemble_hic.analysis_functions import load_samples_from_cfg
from ensemble_hic.setup_functions import make_posterior, parse_config_file

n_structures = [1, 2, 3, 4, 7, 10, 20, 30, 40, 50]
common_path = '/scratch/scarste/ensemble_hic/bau2011/GM12878_new_smallercd_nosphere_'
output_dirs = ['it2_1structures_sn_139replicas',
               'it2_2structures_sn_120replicas',
               'it2_3structures_sn_108replicas',
               'it2_4structures_sn_103replicas',
               'it2_7structures_sn_111replicas',
               'it2_10structures_sn_113replicas',
               '20structures_sn_122replicas',
               '30structures_sn_122replicas',
               '40structures_sn_122replicas',
               'it2_50structures_sn_122replicas',
               ]
output_dirs = [common_path + x + '/' for x in output_dirs]
config_files_GM12878 = [op_dir + 'config.cfg' for op_dir in output_dirs]

common_path = '/scratch/scarste/ensemble_hic/bau2011/K562_new_smallercd_nosphere_'
output_dirs = ['1structures_sn_136replicas',
               '2structures_sn_108replicas',
               '3structures_sn_91replicas',
               '4structures_sn_90replicas',
               '7structures_sn_94replicas',
               '10structures_sn_98replicas',
               '20structures_sn_112replicas',
               '30structures_sn_109replicas',
               '40structures_sn_109replicas',
               'it2_50structures_sn_109replicas',
               ]
output_dirs = [common_path + x + '/' for x in output_dirs]
config_files_K562 = [op_dir + 'config.cfg' for op_dir in output_dirs]

labels = ('GM12878', 'K562')

fig = plt.figure()
ax = fig.add_subplot(111)
bla = []
for i, cfg_files in enumerate((config_files_GM12878, config_files_K562)):
    mean_logLs = []
    mean_logNBs = []
    bla.append([])
    for cfg_file in cfg_files:
        p = make_posterior(parse_config_file(cfg_file))
        logL = lambda x: p.likelihoods['ensemble_contacts'].log_prob(**x.variables)
        logNB = lambda x: p.priors['nonbonded_prior'].log_prob(structures=x.variables['structures'])
        samples = load_samples_from_cfg(cfg_file)
        mean_logLs.append(np.mean(map(logL, samples)))
        # mean_logNBs.append(np.mean(map(logNB, samples)))
        bla[-1].append(np.mean(map(logNB, samples)))
        
    mean_logLs = np.array(mean_logLs)
    ax.plot(n_structures, -mean_logLs,# + mean_logLs.max()+1,
            label=labels[i],
            marker='o')
ax.set_xlabel('# conformers')
ax.set_ylabel('-<log-likelihood> (shifted)')
ax.legend()
#ax.semilogy()

plt.show()
        
