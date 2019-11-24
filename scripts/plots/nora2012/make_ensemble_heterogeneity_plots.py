import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/nora2012/'))

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

from ensemble_quantities import plot_avg_rg_trace
plot_avg_rg_trace(ax1, "plot_data/rogs.pickle")

from TAD_analysis_PNAS import plot_TAD_boundary_hists
plot_TAD_boundary_hists(ax2, "plot_data/TAD_boundaries.pickle",
                        ("plot_data/ENCFF001ZQW.broadPeak.bed",
                         "plot_data/ENCFF001ZQX.broadPeak.bed"))

from plot_intermingling_densities import plot_overlap
plot_overlap(ax3, "plot_data/overlaps.pickle")

fig.tight_layout()
plt.show()
