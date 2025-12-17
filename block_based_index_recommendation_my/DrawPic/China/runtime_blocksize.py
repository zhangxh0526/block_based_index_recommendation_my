import random

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt



datasetnum = 4
indexnum = 4

plt.figure(figsize=(6, 3))
sizel = 12
n = 4
x_ticks = []
xs1 = [[] for i in range(indexnum)]
bar_width = 20
for i in range(datasetnum):#四种选择度
    start = i * bar_width * (indexnum + 1)
    tempx = [start + j * bar_width for j in range(indexnum)]
    print(max(tempx))
    for j in range(len(tempx)):
        xs1[j].append(tempx[j])
    if indexnum % 2 == 1:
        x_ticks.append(tempx[int(indexnum/2)])
    else:
        x_ticks.append((tempx[int(indexnum/2)] + tempx[int(indexnum/2)-1])/2)

x_names = ["10M", "5M", "1M", "0.5M"]

Extend = [75.39, 72.75, 67.44, 70]
Slalom = [89.45, 87.40, 93.39, 89]
Bislearner = [73.39, 61.3, 47, 63]
Swirl = [77.05, 69.75, 64.2, 75.4]


plt.bar(xs1[0], Extend, alpha=0.9, width = bar_width, label='Extend', color='#696969')
plt.bar(xs1[1], Slalom, alpha=0.9, width = bar_width, label='Slalom', color='#1f77b4')
plt.bar(xs1[2], Swirl, alpha=0.9, width = bar_width, label='Swirl(using BIS Act)', color='#76da91')
plt.bar(xs1[3], Bislearner, alpha=0.9, width = bar_width, label='BISlearner', color='#ff7f0e')

plt.legend(loc=(1/23.8, 90.1/100), ncol=4, fontsize=sizel-3.8, frameon=False)
# plt.legend()
# plt.title("wiki_pagecount", fontsize=sizel)
plt.ylim((30, 100))
plt.xticks(x_ticks, x_names, rotation=0, fontsize=sizel-2)
# plt.yticks([40, 60, 80, 100], [40, 60, 80, 100], fontsize=sizel)
plt.minorticks_off()
plt.ylabel("Relative workload cost \n (% of without indexes)", fontsize=sizel-1, fontweight='bold')
plt.xlabel('Records num per block', fontsize=11, fontweight='bold')

plt.tight_layout()

plt.savefig("block_size.pdf", dpi=300, bbox_inches='tight')
plt.show()
