import sys
import numpy as np
from ensemble_hic.setup_functions import parse_config_file, make_posterior
from ensemble_hic.analysis_functions import load_sr_samples

cfg_file = sys.argv[1]
max_samples = int(sys.argv[2])
burnin = int(sys.argv[3])

settings = parse_config_file(cfg_file)
n_replicas = int(settings['replica']['n_replicas'])
samples = load_sr_samples(settings['general']['output_folder'] + 'samples/',
                          n_replicas, max_samples, 1000, burnin)
posterior = make_posterior(settings)
Es = [-posterior.log_prob(**x.variables) for x in samples]

print "Mean negative log-probability: ", np.mean(Es)
