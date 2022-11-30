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

real_streams = [
    'covtypeNorm-1-2vsAll',
    'electricity',
    'poker-lsn-1-2vsAll',
    'INSECTS-abrupt',
    'INSECTS-gradual',
    'INSECTS-incremental'
    ]

base_clfs = ['GNB', 'KNN', 'SVM', 'DT', 'MLP']

n_drift_types=3
stream_reps=5

res = np.load('res_clf_cls/real_clf.npy') # measures, datasets, reps, folds, clfs

res_mean = np.mean(res, axis=2)
print(res_mean.shape)

fig, ax = plt.subplots(2, 3, figsize=(13,11), sharex=True, sharey=True)
ax=ax.ravel()
plt.suptitle('Real-world', fontsize=18)

for dataset_id, dataset in enumerate(real_streams):
        
    axx = ax[dataset_id]
    
    r = res_mean[:,dataset_id]
    axx.imshow(r, vmin=0.05, vmax=1.)
    
    for _a, __a in enumerate(measures):
        for _b, __b in enumerate(base_clfs):
            axx.text(_b, _a, "%.3f" % (r[_a, _b]) , va='center', ha='center', c='white', fontsize=11)

    axx.set_title(dataset)
    axx.set_xticks(np.arange(len(base_clfs)),base_clfs)
    axx.set_yticks(np.arange(len(measures)),measures)
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_clf/real.png')
