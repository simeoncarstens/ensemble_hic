import os
import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook

dpath = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/nature11049-s5.xls')
wb = open_workbook(dpath)
sheet = wb.sheets()[0]

table = np.array([np.array(sheet.row_values(j))[list((3,4,8)) +
                                                range(10, sheet.ncols)]
                  for j in range(3, sheet.nrows)])

region_start = 100378306
region_end = 101298738

in_region = np.array(filter(lambda x: float(x[0]) >= region_start
                            and float(x[1]) <= region_end, table))


fig, ax = plt.subplots()
space = range(len(in_region))
ax.plot(space, in_region[:,3], ls='--', marker='x')
ax.set_xticks(space)
ax.set_xticklabels(in_region[:,2])
ax.set_ylabel('log2 transcript level')
ax.set_xlabel('gene')
ax.axvline(5.5, c='r')

plt.show()
