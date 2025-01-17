"""
Plot.
E3, E4 - Visualize -- select k-best + classification + f-test anova --- Real-world streams
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
from matplotlib.lines import Line2D


base_clfs = ['GNB','KNN','SVM','DT','MLP']

real_streams = [
    'covtypeNorm-1-2vsAll',
    'electricity',
    'poker-lsn-1-2vsAll',
    'INSECTS-abrupt',
    'INSECTS-gradual',
    'INSECTS-incremental'
    ]

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]


# CLF
fig, ax = plt.subplots(3,2,figsize=(10,7), sharex=True)
c = plt.cm.turbo(np.linspace(0,1,6))
ax = ax.ravel()

for dataset_id, dataset in enumerate(real_streams):
    
    clf = np.load('results/clf_sel_real_%i.npy' % dataset_id)
    clf_mean = np.mean(clf, axis=1)
    
    for cm_id, cm in enumerate(clf_mean.T):
        ax[dataset_id].plot(n_features, cm, label=base_clfs[cm_id], c=c[cm_id])
    ax[dataset_id].set_title(dataset)    
    ax[dataset_id].set_xticks(np.arange(len(n_features)),n_features)
    ax[dataset_id].spines['top'].set_visible(False)
    ax[dataset_id].spines['right'].set_visible(False)
    ax[dataset_id].grid(ls=':')
    ax[dataset_id].set_ylabel('balanced accuracy score')
    ax[dataset_id].set_xlabel('n features')
    
    ax[dataset_id].set_xticks(n_features, 
                [('%i' % s) if ss%2==0 else '' for ss, s in enumerate(n_features)])

    if dataset_id==3:
        ax[dataset_id].legend(ncol=2, frameon=False)

        
plt.tight_layout()
plt.savefig('figures/fig_clf/sel_real.png')    
plt.savefig('figures/fig_clf/sel_real.eps')    
plt.clf()

# ANOVA

anovas = []
for dataset_id, dataset in enumerate(real_streams):
    
    anova = np.load('results/anova_sel_real_%i.npy' % dataset_id)
    anovas.append(anova[:,0])


anovas = np.array(anovas)
anova_sum = np.nansum(anovas, axis=0)
sort_order = np.flip(np.argsort(anova_sum))

labels_measures = utils.measure_labels_selected
labels_counts = [len(l) for l in labels_measures]
labels_ids = [[c_id for _ in range(cnt)] for c_id,cnt in enumerate(labels_counts)]
labels_ids = np.array(sum(labels_ids, []))[sort_order]

labels_measures = np.array(sum(labels_measures, []))

cols=c

fig, ax = plt.subplots(1,1,figsize=(9,9/1.618), sharex=True, sharey=True)

start = np.zeros_like(anovas[0])
for dataset_id, dataset in enumerate(real_streams): 
    temp = anovas[dataset_id]
    l = labels_measures[sort_order]
    t = temp[sort_order]
    ax.bar(range(len(l)), t, bottom=start, alpha=((1/(len(real_streams)+1))*(dataset_id+1)), color=cols[labels_ids])
    t[np.isnan(t)] = 0
    start+=t
    
ax.set_title('Real data streams')
ax.set_ylabel('accumulated F-statistic')
    
ax.set_xticks(range(len(l)),l,rotation=45, ha='right', fontsize=8)
ax.grid(ls=":")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

custom_lines = [Line2D([0], [0], color=cols[0], lw=4),
                Line2D([0], [0], color=cols[1], lw=4),
                Line2D([0], [0], color=cols[2], lw=4),
                Line2D([0], [0], color=cols[3], lw=4),
                Line2D([0], [0], color=cols[4], lw=4)]
ax.legend(custom_lines, ['Clustering', 'Complexity', 'Info theory', 'Landmarking', 'Statistical'], ncol=3, frameon=False)
ax.set_xlim(-1,50-0.5)

plt.tight_layout()
plt.savefig('figures/fig_clf/anova_real.png')
plt.savefig('figures/fig_clf/anova_real.eps')

# REDUCED

fig, ax = plt.subplots(3, 2, figsize=(8,8), sharex=True, sharey=True)
ax = ax.ravel()
res = np.load('results/real_clf_reduced.npy')

for f_id, f in enumerate(real_streams):
    clf = np.load('results/clf_sel_real_%i.npy' % f_id)    

    img = np.zeros((2,5))
    reduced = np.mean(res[f_id], axis=0)
    full = np.mean(clf[-1], axis=0)
    img[0] = full
    img[1] = reduced-full
    
    ax[f_id].imshow(img, vmin=0.05, vmax=1, cmap='Blues')
    ax[f_id].set_title(f)
    
    ax[f_id].set_xticks(range(len(base_clfs)), base_clfs)
    ax[f_id].set_yticks(range(2), ['full', 'reduced'])
    
    for _a, __a in enumerate(['full', 'reduced']):
        for _b, __b in enumerate(base_clfs):
            if _a==0:
                ax[f_id].text(_b, _a, "%.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
            else:
                ax[f_id].text(_b, _a, "%+.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
    

plt.tight_layout()
plt.savefig('figures/fig_clf/reduced_real.png')
plt.savefig('baz.png')