"""
Visualization for E4 - MOA
"""
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import os

base_clfs = ['GNB','KNN','SVM','DT','MLP']

streams = os.listdir('data/moa')
streams.remove('.DS_Store')

fig, ax = plt.subplots(4,1,figsize=(10, 12))

margin = .3
w = .05

c = plt.cm.turbo(np.linspace(0,1,6))
c2 = np.copy(c)
c2[:,:3] = c2[:,:3]/2+.5
print(c)

legend_labels=[ '%s-%s' % (bc, d) for bc in base_clfs for d in ['F', 'R']]
legend_colors=[]
for i in range(len(c)):
    legend_colors.append(c[i])
    legend_colors.append(c2[i])
custom_lines = [Line2D([0], [0], color=legend_colors[i], lw=4) for i in range(len(legend_colors))]
ax[-1].legend(custom_lines, legend_labels, ncol=5, frameon=False,  fontsize=11)

row=0

res = np.load('results/moa_clf_reduced.npy')
for f_id, f in enumerate(streams):
            
    clf = np.load('results/clf_sel_moa_%i.npy' % f_id)    

    reduced = np.mean(res[f_id], axis=0)
    full = np.mean(clf[-1], axis=0)
    

    x = np.linspace(-margin, margin, len(full)) + f_id%3

    ax[row].bar(x-w/2, full, width=w, color=c)
    ax[row].bar(x+w/2, reduced, width=w, color=c2)
    
    ax[row].set_xticks(np.arange(3), streams[row*3:(row+1)*3])
    
    if f_id%3==2:
        row+=1
    
# Styling
ax[0].set_title('RBF')
ax[1].set_title('LED')
ax[2].set_title('HYPERPLANE')
ax[3].set_title('SEA')
ax[0].set_xlim(ax[-1].get_xlim())

for aids, aa in enumerate(ax.ravel()):
    aa.set_ylim(0,1)
    aa.grid(axis='y', ls=':')
    
    aa.set_ylabel('balanced accuracy score')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/reduced_moa.png')
plt.savefig('figures/reduced_moa.eps')
