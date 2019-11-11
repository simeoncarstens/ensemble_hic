import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/'))

fig_shape = (5,6)

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/nora2012/'))
fish_axes = [plt.subplot2grid(fig_shape, (i / 3, (i % 3) * 2), colspan=2)
             for i in range(7)]
from distance_distributions import plot_all_hists
plot_all_hists(fish_axes, "plot_data/distance_distributions.pickle")

from differentiation import plot_before_hist, plot_after_hist
before_ax = plt.subplot2grid(fig_shape, (3, 0), rowspan=2, colspan=3)
plot_before_hist(before_ax)
after_ax = plt.subplot2grid(fig_shape, (3, 3), rowspan=2, colspan=3,
                            sharex=before_ax)
plot_after_hist(after_ax)

plt.gcf().tight_layout()
