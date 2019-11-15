import sys
from cPickle import dump
from ensemble_hic.setup_functions import parse_config_file
from ensemble_hic.analysis_functions import load_sr_samples

settings = parse_config_file(sys.argv[1])
output_folder = settings['general']['output_folder']

# just to be sure that we ship the settings used for the
# simulation which yielded the samples
settings = parse_config_file(output_folder + 'config.cfg')

max_samples = 50001
burnin = 30000
samples = load_sr_samples(output_folder + 'samples/',
                          int(settings['replica']['n_replicas']),
                          max_samples, 1000, burnin) 
with open(sys.argv[2], "w") as opf:
    dump(samples, opf)
