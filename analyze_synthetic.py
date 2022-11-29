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
res_ext = np.load('res_clf_cls/clf_ext.npy') # measures, drifts, reps, folds, clfs
print(res.shape)

res_mean = np.mean(res, axis=(2,3))
res_ext_mean = np.mean(res_ext, axis=(2,3))
print(res_mean.shape)

fig, ax = plt.subplots(2, 3, figsize=(13,12))
plt.suptitle('Synthetic', fontsize=18)

for r_id, r in enumerate([res_mean, res_ext_mean]):
    for drift_type_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']):
        ax[r_id,drift_type_id].imshow(r[:,drift_type_id], vmin=0.05, vmax=1.)
        
        for _a, __a in enumerate(measures):
            for _b, __b in enumerate(base_clfs):
                ax[r_id,drift_type_id].text(_b, _a, "%.3f" % (r[:,drift_type_id][_a, _b]) , va='center', ha='center', c='white', fontsize=11)
        
        ax[r_id,drift_type_id].set_xticks(np.arange(len(base_clfs)),base_clfs)
        ax[r_id,drift_type_id].set_yticks(np.arange(len(measures)),measures)
        if r_id==1:
            ax[r_id,drift_type_id].set_title(drift_type+' extraction')
        else:
            ax[r_id,drift_type_id].set_title(drift_type)
        
plt.tight_layout()
plt.savefig('foo.png')
