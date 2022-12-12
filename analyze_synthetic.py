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

base_clfs = ['GNB', 'KNN', 'SVM', 'DT', 'MLP']

n_drift_types=3
stream_reps=5

res = np.load('res_clf_cls/clf.npy') # measures, drifts, reps, folds, clfs
print(res.shape)

res_mean = np.mean(res, axis=(2,3))
print(res_mean.shape)

fig, ax = plt.subplots(1, 3, figsize=(13,8), sharex=True, sharey=True)
plt.suptitle('Synthetic', fontsize=18)

r = res_mean
for drift_type_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']):
    ax[drift_type_id].imshow(r[:,drift_type_id], vmin=0.05, vmax=1., cmap='Blues')
    
    for _a, __a in enumerate(measures):
        for _b, __b in enumerate(base_clfs):
            ax[drift_type_id].text(_b, _a, "%.3f" % (r[:,drift_type_id][_a, _b]) , va='center', ha='center', c='black' if r[:,drift_type_id][_a, _b]<0.5 else 'white', fontsize=11)
    
    ax[drift_type_id].set_xticks(np.arange(len(base_clfs)),base_clfs)
    ax[drift_type_id].set_yticks(np.arange(len(measures)),measures)
    ax[drift_type_id].set_title(drift_type)
        
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_clf/syn.png')
