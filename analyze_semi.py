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

res = np.load('res_clf_cls/semi_clf.npy') # measures, datasets, drifts, reps, folds, clfs
res = res[:,:,:2]

res_mean = np.mean(res, axis=(3))
print(res_mean.shape)

fig, ax = plt.subplots(2, 6, figsize=(20,10), sharex=True, sharey=True)
plt.suptitle('Semi-synthetic', fontsize=18)

for dataset_id, dataset in enumerate(static_data):
    for drift_type_id, drift_type in enumerate(['Nearest', 'Cubic']):
        
        axx = ax[drift_type_id, dataset_id]
        
        r = res_mean[:,dataset_id,drift_type_id]
        axx.imshow(r, vmin=0.05, vmax=1., cmap='Blues')
        
        for _a, __a in enumerate(measures):
            for _b, __b in enumerate(base_clfs):
                axx.text(_b, _a, "%.3f" % (r[_a, _b]) , va='center', ha='center', c='black' if r[_a, _b]<0.5 else 'white', fontsize=11)
        
        if drift_type_id==0:
            axx.set_title(dataset)
        else:
            axx.set_xticks(np.arange(len(base_clfs)),base_clfs)

        if dataset_id==0:
            axx.set_ylabel(drift_type)
            axx.set_yticks(np.arange(len(measures)),measures)
        
        
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_clf/semi.png')
