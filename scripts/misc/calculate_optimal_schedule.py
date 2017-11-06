import sys, os, numpy as np
from cPickle import dump

from ensemble_hic.setup_functions import parse_config_file

pypath = os.path.expanduser('~/projects/adarex/py')
if not pypath in sys.path: sys.path.insert(0, pypath)
from scheduler import Scheduler, RelativeEntropy, SwapRate, SimpleScheduler, SwapRate
from csbplus.statmech.ensembles import BoltzmannEnsemble

dos = np.load(sys.argv[1])
target_entropy = float(sys.argv[2])
variables = sys.argv[3]
output_file = sys.argv[4]

from csbplus.statmech.dos import DOS
#ensemble = BoltzmannEnsemble(dos=dos)
ensemble = BoltzmannEnsemble(dos=DOS(dos.E.sum(1), dos.s))
entropy  = Scheduler(ensemble, RelativeEntropy(), np.greater)
entropy.find_schedule(target_entropy, 1e-6, 1., verbose=True)

beta = np.array(entropy.schedule)
beta[0] = 1e-6
beta[-1] = 1.

if False:
    from scheduler import SwapRate, SimpleScheduler
    step = 1
    tempsched = SimpleScheduler(ensemble, SwapRate(), comparison=np.less)
    pred_swap_rates = [tempsched.eval_criterion(beta[i], beta[i+1])
                       for i in range(len(beta)-1)[::step]]

    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot((beta[1:] + beta[:-1])[::step] / 2.0, pred_swap_rates)
        ax.set_xlabel('beta')
        ax.set_ylabel('predicted swap rate')
        plt.show()

print "Schedule length:", len(beta)
with open(output_file,'w') as opf:
    schedule = {}
    for var in variables.split(','):
        schedule.update(**{var: beta})
    dump(schedule, opf)
