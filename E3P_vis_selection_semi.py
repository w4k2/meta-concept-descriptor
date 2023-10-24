"""
Plot.

E3, E4 - selekcja k-best + ANOVA - semi
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
from matplotlib.lines import Line2D


base_clfs = ['GNB','KNN','SVM','DT','MLP']

static_data = ['australian',
    'banknote',
    'diabetes',
    'german',
    'vowel0',
    'wisconsin'
    ]

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]

n_drift_types = 3
stream_reps = 5

clf = np.load('res_clf_cls/clf_sel_semi.npy')
anova = np.load('res_clf_cls/anova_sel_semi.npy')

print(clf.shape) # drfs, datasets, features, folds, clfs
print(anova.shape) # drfs, datasets, features, (stat, val)

# CLF
fig, ax = plt.subplots(3,2,figsize=(10,7), sharex=True)
c = plt.cm.turbo(np.linspace(0,1,6))
ax=ax.ravel()

for dataset_id, dataset in enumerate(static_data):
    axx = ax[dataset_id]

    clf_temp = clf[dataset_id]
    print(clf_temp.shape)
    # exit()
    clf_temp_mean = np.mean(clf_temp, axis=(0,2))
    print(clf_temp_mean.shape)
    
    for cm_id, cm in enumerate(clf_temp_mean.T):
        axx.plot(n_features, cm, label=base_clfs[cm_id], c=c[cm_id])
        
    axx.set_title(dataset)  
    axx.set_xticks(n_features, 
                   [('%i' % s) if ss%2==0 else '' for ss, s in enumerate(n_features)])
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)
    axx.grid(ls=':')
    axx.set_ylabel('blanced accuracy')
    axx.set_xlim(*n_features[::(len(n_features)-1)])
    #axx.set_xlabel('n features')

    #if dataset_id==0:
    #    axx.legend()
        
    if dataset_id==1:
        axx.legend(ncol=3, frameon=False, loc=8)
    
    if dataset_id == 5:
        axx.set_xlabel('number of features')

plt.tight_layout()
plt.savefig('fig_clf/sel_semi.png')
plt.savefig('fig_clf/sel_semi.eps')
plt.savefig('foo.png')    
plt.clf()

# ANOVA

anova_sum = np.nansum(anova[:,:,:,0], axis=(0,1))
sort_order = np.flip(np.argsort(anova_sum))

labels_measures = utils.measure_labels_selected
labels_counts = [len(l) for l in labels_measures]
labels_ids = [[c_id for _ in range(cnt)] for c_id,cnt in enumerate(labels_counts)]
labels_ids = np.array(sum(labels_ids, []))[sort_order]

labels_measures = np.array(sum(labels_measures, []))

cols=c

fig, ax = plt.subplots(2,1,figsize=(12,12/1.618), sharex=True, sharey=True)

for d_id, drift_type in enumerate(['Nearest', 'Cubic']):   
    start = np.zeros_like(anova[dataset_id,d_id,:,0])
    for dataset_id, dataset in enumerate(static_data): 
        ax[d_id].set_title(drift_type)    
        temp = anova[dataset_id,d_id,:,0]
        l = labels_measures[sort_order]
        t = temp[sort_order]
        ax[d_id].bar(range(len(l)), t, bottom=start, alpha=((1/(len(static_data)+1))*(dataset_id+1)), color=cols[labels_ids])
        t[np.isnan(t)] = 0
        start+=t
    
    ax[d_id].set_xticks(range(len(l)),l,rotation=45, ha='right', fontsize=8)
    ax[d_id].grid(ls=":")
    ax[d_id].spines['top'].set_visible(False)
    ax[d_id].spines['right'].set_visible(False)
    ax[d_id].set_xlim(-1,50-0.5)
    ax[d_id].set_yscale('log')
    ax[d_id].set_ylim(1, 1000000)
    ax[d_id].set_ylabel('accumulated F-statistic (log scale)')
    
custom_lines = [Line2D([0], [0], color=cols[0], lw=4),
                Line2D([0], [0], color=cols[1], lw=4),
                Line2D([0], [0], color=cols[2], lw=4),
                Line2D([0], [0], color=cols[3], lw=4),
                Line2D([0], [0], color=cols[4], lw=4)]
ax[0].legend(custom_lines, ['Clustering', 'Complexity', 'Info theory', 'Landmarking', 'Statistical'], ncol=3, frameon=False)
        
plt.tight_layout()
plt.savefig('fig_clf/anova_semi.png')
plt.savefig('fig_clf/anova_semi.eps')
plt.savefig('bar.png')

# REDUCED
reduced = np.load('res_clf_cls/semi_clf_reduced.npy')
print(reduced.shape) # 6, 2, 10, 5
# exit()

reduced_mean = np.mean(reduced, axis=2)

fig, ax = plt.subplots(6, 2, figsize=(8,12), sharex=True, sharey=True)

for dataset_id, dataset in enumerate(static_data):
    for drf_id, drift_type in enumerate(['nearest', 'cubic']):    
        img = np.zeros((2,5))
        full = np.mean(clf[dataset_id, drf_id, -1], axis=0)
        reduced = reduced_mean[dataset_id,drf_id]
        img[0] = full
        img[1] = reduced-full
        
        ax[dataset_id,drf_id].imshow(img, vmin=0.05, vmax=1, cmap='Blues')
        ax[dataset_id,drf_id].set_title('%s %s' % (dataset, drift_type))
        
        ax[dataset_id,drf_id].set_xticks(range(len(base_clfs)), base_clfs)
        ax[dataset_id,drf_id].set_yticks(range(2), ['full', 'reduced'])
        
        for _a, __a in enumerate(['full', 'reduced']):
            for _b, __b in enumerate(base_clfs):
                if _a==0:
                    ax[dataset_id,drf_id].text(_b, _a, "%.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
                else:
                    ax[dataset_id,drf_id].text(_b, _a, "%+.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
    
plt.tight_layout()
plt.savefig('fig_clf/reduced_semi.png')
plt.savefig('baz.png')
