"""
Plot E2 - Visualize classification results for MOA streams
"""
import numpy as np
import matplotlib.pyplot as plt
import os

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

streams = os.listdir('data/moa')
streams.remove('.DS_Store')
print(streams)

base_clfs = ['GNB', 'KNN', 'SVM', 'DT', 'MLP']

n_drift_types=3
stream_reps=5

res = np.load('results/moa_clf.npy') # measures, datasets, reps, folds, clfs

res_mean = np.mean(res, axis=2)
print(res_mean.shape)

fig, ax = plt.subplots(4, 3, figsize=(10,20), sharex=True, sharey=True)
ax=ax.ravel()
plt.suptitle('MOA', fontsize=18, y=0.99)

for dataset_id, dataset in enumerate(streams):
        
    axx = ax[dataset_id]
    
    r = res_mean[:,dataset_id]
    axx.imshow(r, vmin=0.4, vmax=1., cmap='Blues')
    
    for _a, __a in enumerate(measures):
        for _b, __b in enumerate(base_clfs):
            axx.text(_b, _a, "%.3f" % (r[_a, _b]) , va='center', ha='center', c='black' if r[_a, _b]<0.75 else 'white', fontsize=11)

    axx.set_title(dataset.split('.')[0])
    axx.set_xticks(np.arange(len(base_clfs)),base_clfs)
    axx.set_yticks(np.arange(len(measures)),measures)
    

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/fig_clf/MOA.png')
