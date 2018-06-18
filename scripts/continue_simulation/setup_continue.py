import sys
import os
import numpy as np

from ensemble_hic.setup_functions import parse_config_file

sys.argv = ['asdfasdf',
            '/scratch/scarste/ensemble_hic/nora2012/bothdomains_it3_rep4_20structures_309replicas/config.cfg',
            80001,
            1]

config_file = sys.argv[1]
n_samples = int(sys.argv[2])
n_cont = int(sys.argv[3])

settings = parse_config_file(config_file)
output_folder = settings['general']['output_folder']
cont_folder = output_folder + 'init_continue{}/'.format(n_cont)
if not os.path.exists(cont_folder):
    os.makedirs(cont_folder)
samples_folder = output_folder + 'samples/'
n_replicas = int(settings['replica']['n_replicas'])
fname = samples_folder + 'samples_replica{}_{}-{}.pickle'
dump_interval = int(settings['replica']['samples_dump_interval'])

## determine offset
offset = 0
while True:
    if os.path.exists(fname.format(1, offset, offset + dump_interval)):
        offset += dump_interval
    else:
        break
settings['replica'].update(offset=offset)

if not os.path.exists(cont_folder):
    os.makedirs(cont_folder)

## assemble and write start states
start_states = [np.load(fname.format(i, offset - dump_interval, offset))[-1].variables['structures']
                for i in range(1, n_replicas + 1)]
np.save(cont_folder + 'init_states.npy', np.array(start_states))

if n_cont == 1:
    mcmc_stats = np.loadtxt(output_folder + 'statistics/mcmc_stats.txt')
else:
    mcmc_stats = np.loadtxt(output_folder + 'init_continue{}/statistics/mcmc_stats.txt'.format(n_cont - 1))
    
## assemble and write HMC timesteps
timesteps = mcmc_stats[-1,2::2]
np.save(cont_folder + 'timesteps.npy', timesteps)

## write continue config file
settings['structures_hmc']['timestep'] = cont_folder + 'timesteps.npy'
settings['structures_hmc']['initial_state'] = cont_folder + 'init_states.npy'
settings['replica']['n_samples'] = n_samples
settings['replica'].update(stats_folder=cont_folder + 'statistics/')
settings['general'].update(cont_folder=cont_folder)
with open(cont_folder + 'cont_config.cfg', 'w') as opf:
    for section_name, params in settings.iteritems():
        opf.write('[{}]\n'.format(section_name))
        for k, v in params.iteritems():
            opf.write('{} = {}\n'.format(k, v))
        opf.write('\n')
