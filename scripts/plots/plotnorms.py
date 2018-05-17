import os
import numpy as np
from ensemble_hic.analysis_functions import load_samples
from ensemble_hic.setup_functions import make_posterior, parse_config_file
os.chdir('/scratch/scarste/ensemble_hic/hairpin_s/hairpin_s_fwm_poisson_fwm_poisson_2structures_sn_50replicas')
#os.chdir('/scratch/scarste/ensemble_hic/proteins/1pga_1shf_fwm_poisson_2structures_sn_80replicas')

s = load_samples('samples/', 50, 50001,1000,40000)
pacc=np.loadtxt('statistics/re_stats.txt')
norms = np.array([[x.variables['norm'] for x in y] for y in s])

import matplotlib.pyplot as plt

cfg = parse_config_file('config.cfg')
p = make_posterior(cfg)
shape = float(cfg['norm_prior']['shape'])
rate = float(cfg['norm_prior']['rate'])
L = p.likelihoods['ensemble_contacts']
A = L.error_model.data.sum()
B = lambda X: L.forward_model(norm=1.0, structures=X).sum()
Bs = np.array([[B(x.variables['structures']) for x in y] for y in s])
lammdas = np.load('schedule.pickle')['lammda']
mean_gammas = np.array([(lammdas[i] * A + shape)/(lammdas[i] * Bs[i].mean() + rate) for i in range(len(s))])
var_gammas = np.array([((lammdas[i] * A + shape)/(lammdas[i] * Bs[i] + rate)**2).mean() for i in range(len(s))])
var_gammas2 = np.array([[((lammdas[i] * A + shape)/(lammdas[i] * Bs[i,j] + rate)**2) for j in range(s.shape[1])] for i in range(len(s))])


plt.figure()
plt.plot(lammdas,var_gammas2.mean(1), label='variance (theory)')
plt.plot(lammdas,norms.var(1), label='variance (samples)')
plt.legend()
plt.semilogx()
plt.xlabel('lambda')
plt.ylabel('variance(gamma)')
plt.show()


plt.figure()
plt.plot(lammdas, norms.mean(1), label='mean (samples)', linewidth='6')
plt.plot(lammdas, mean_gammas,label='mean (theory)')
plt.legend()
plt.semilogx()
plt.xlabel('lambda')
plt.ylabel('mean(gamma)')
plt.show()


gammas = np.array([[(lammdas[i] * A + 0.1)/(lammdas[i] * Bs[i,j] + 0.1) for j in range(s.shape[1])]for i in range(len(s))])
