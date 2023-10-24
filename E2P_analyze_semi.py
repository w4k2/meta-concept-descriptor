"""
Plot E2 - semi
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1233)

measures = ["clustering",
    "complexity",
    "concept",
    "general",
    "info-theory",
    "itemset",
    "landmarking",
    "model-based",
    "statistical"
    ]

static_data = ['australian',
    'banknote',
    'diabetes',
    'german',
    'vowel0',
    'wisconsin'
    ]

base_clfs = ['GNB', 'KNN', 'SVM', 'DT', 'MLP']

n_drift_types=3
stream_reps=5

res = np.load('results/semi_clf.npy') # measures, datasets, drifts, reps, folds, clfs
res = res[:,:,:2]

res_mean = np.mean(res, axis=(2,3))

fig, ax = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)
ax = ax.ravel()
plt.suptitle('Semi-synthetic', fontsize=18)

for dataset_id, dataset in enumerate(static_data):
        
    axx = ax[dataset_id]
    
    r = res_mean[:,dataset_id]
    axx.imshow(r, vmin=0.05, vmax=1., cmap='Blues')
    
    for _a, __a in enumerate(measures):
        for _b, __b in enumerate(base_clfs):
            axx.text(_b, _a, "%.3f" % (r[_a, _b]) , va='center', ha='center', c='black' if r[_a, _b]<0.5 else 'white', fontsize=11)
    
    axx.set_title(dataset)
    axx.set_xticks(np.arange(len(base_clfs)),base_clfs)

    if dataset_id==0:
        axx.set_yticks(np.arange(len(measures)),measures)
        
        
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/fig_clf/semi.png')
