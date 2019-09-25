import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/'))

fig_shape = (6,9)

os.chdir(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/'))
ev_ax = plt.subplot2grid(fig_shape, (0,0), rowspan=3, colspan=3)
from evidences import plot_evidences
plot_evidences(ev_ax)

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/nora2012/'))
fish_axes = [plt.subplot2grid(fig_shape, (i / 3, 3 + i % 3)) for i in range(7)]
from distance_distributions import plot_all_hists
plot_all_hists(fish_axes)

from TAD_analysis import plot_TADcm_hist
mix_ax = plt.subplot2grid(fig_shape, (3, 3), rowspan=3, colspan=3)
plot_TADcm_hist(mix_ax)

from differentiation import plot_before_hist, plot_after_hist
before_ax = plt.subplot2grid(fig_shape, (0, 6), rowspan=3, colspan=3)
plot_before_hist(before_ax)
after_ax = plt.subplot2grid(fig_shape, (3, 6), rowspan=3, colspan=3,
                            sharex=before_ax)
plot_after_hist(after_ax)

plt.gcf().tight_layout()
plt.gcf().set_size_inches((12.24, 6.01))
plt.gcf().subplots_adjust(left=0.12, bottom=0.13,
                          right=0.96, top=0.96,
                          wspace=0.75, hspace=.95)
