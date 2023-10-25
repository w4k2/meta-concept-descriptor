"""
Plot.

E3, E4 - Visualize -- select k-best + classification + f-test anova --- Semi-synthetic streams
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
from matplotlib.lines import Line2D


base_clfs = ['GNB','KNN','SVM','DT','MLP']

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]

n_drift_types = 3
stream_reps = 5

clf = np.load('results/clf_sel.npy')
anova = np.load('results/anova_sel.npy')

print('A', clf.shape) # drfs, reps, features, folds, clfs
print(anova.shape) # drfs, reps, features, (stat, val)

# CLF
fig, ax = plt.subplots(3,1,figsize=(10,7), sharex=True)
c = plt.cm.turbo(np.linspace(0,1,6))

for d_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']):    
    clf_temp = clf[d_id]
    clf_temp_mean = np.mean(clf[d_id], axis=(0,2))
        
    for cm_id, cm in enumerate(clf_temp_mean.T):
        ax[d_id].plot(n_features, cm, label=base_clfs[cm_id], c=c[cm_id])
    ax[d_id].set_title('%s drifts' % drift_type)    
    ax[d_id].set_xticks(n_features)
    ax[d_id].spines['top'].set_visible(False)
    ax[d_id].spines['right'].set_visible(False)
    ax[d_id].grid(ls=':')
    ax[d_id].set_ylabel('balanced accuracy score')
    ax[d_id].set_xlim(*n_features[::(len(n_features)-1)])

    if d_id==0:
        ax[d_id].legend(ncol=3, frameon=False)
    
    if d_id == 2:
        ax[d_id].set_xlabel('number of features')
        
plt.tight_layout()
plt.savefig('figures/fig_clf/sel_syn.png')
plt.savefig('figures/fig_clf/sel_syn.eps')
plt.savefig('foo.png')
    
plt.clf()

"""
# ANOVA
"""
anova_sum = np.nansum(anova[:,:,:,0], axis=(0,1))
sort_order = np.flip(np.argsort(anova_sum))

labels_measures = utils.measure_labels_selected
labels_counts = [len(l) for l in labels_measures]
labels_ids = [[c_id for _ in range(cnt)] for c_id,cnt in enumerate(labels_counts)]
labels_ids = np.array(sum(labels_ids, []))[sort_order]

labels_measures = np.array(sum(labels_measures, []))

cols = c

fig, ax = plt.subplots(3,1,figsize=(12,12/1.618), sharex=True, sharey=True)

for d_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']):    
    ax[d_id].set_title('%s drift' % drift_type)    
    start = np.zeros_like(anova[d_id,0,:,0])
    for r_id in range(stream_reps):
        temp = anova[d_id,r_id,:,0]
        l = labels_measures[sort_order]
        t = temp[sort_order]
        ax[d_id].bar(range(len(l)), t, bottom=start, alpha=((1/(stream_reps+1))*(r_id+1)), color=cols[labels_ids])
        t[np.isnan(t)] = 0
        start+=t
    ax[d_id].set_xticks(range(len(l)),l,rotation=45, ha='right', fontsize=8)
    ax[d_id].grid(ls=":")
    ax[d_id].spines['top'].set_visible(False)
    ax[d_id].spines['right'].set_visible(False)
    ax[d_id].set_ylabel('accumulated F-statistic')
    ax[d_id].set_xlim(-1,50-0.5)

custom_lines = [Line2D([0], [0], color=cols[0], lw=4),
                Line2D([0], [0], color=cols[1], lw=4),
                Line2D([0], [0], color=cols[2], lw=4),
                Line2D([0], [0], color=cols[3], lw=4),
                Line2D([0], [0], color=cols[4], lw=4)]
ax[0].legend(custom_lines, ['Clustering', 'Complexity', 'Info theory', 'Landmarking', 'Statistical'], ncol=3, frameon=False)
        
plt.tight_layout()
plt.savefig('figures/fig_clf/anova_syn.png')
plt.savefig('figures/fig_clf/anova_syn.eps')
plt.savefig('bar.png')

"""
# REDUCED
"""
reduced = np.load('results/clf_reduced.npy')
print(reduced.shape) # 3, 5, 10, 5

reduced_mean = np.mean(reduced, axis=(1,2))

fig, ax = plt.subplots(3, 1, figsize=(8,8), sharex=True, sharey=True)
    
for drf_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']):    
    img = np.zeros((2,5))
    full = np.mean(clf[drf_id, :, -1,:,:], axis=(0,1))
    reduced = reduced_mean[drf_id]
    img[0] = full
    img[1] = reduced-full
    
    ax[drf_id].imshow(img, vmin=0.05, vmax=1, cmap='Blues')
    ax[drf_id].set_title('%s drift' % drift_type)
    
    ax[drf_id].set_xticks(range(len(base_clfs)), base_clfs)
    ax[drf_id].set_yticks(range(2), ['full', 'reduced'])
    
    for _a, __a in enumerate(['full', 'reduced']):
        for _b, __b in enumerate(base_clfs):
            if _a==0:
                ax[drf_id].text(_b, _a, "%.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
            else:
                ax[drf_id].text(_b, _a, "%+.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
    
plt.tight_layout()
plt.savefig('figures/fig_clf/reduced_syn.png')
plt.savefig('baz.png')
