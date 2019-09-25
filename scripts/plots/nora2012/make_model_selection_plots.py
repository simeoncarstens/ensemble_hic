import os
import sys
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/plots'))
os.chdir(os.path.expanduser('~/projects/ensemble_hic/scripts/plots'))
from evidences import plot_evidences
plot_evidences(ax1)

sys.path.append(os.path.expanduser('~/projects/ensemble_hic/scripts/plots/nora2012'))
from md_data_scatter import plot_md_d_scatter
plot_md_d_scatter(ax2)

fig.tight_layout()
plt.show()
