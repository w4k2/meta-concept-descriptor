import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1233)

base_clfs = ['GNB','KNN','SVM','DT','MLP']

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]

n_drift_types = 3
stream_reps = 5

clf = np.load('res_clf_cls/clf_sel.npy')
anova = np.load('res_clf_cls/anova_sel.npy')

print(clf.shape) # drfs, reps, features, folds, clfs
print(anova.shape) # drfs, reps, features, (stat, val)

fig, ax = plt.subplots(1,3,figsize=(10,15), sharex=True, sharey=True)

for d_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']):    
    clf_temp = clf[d_id]
    clf_temp_mean = np.mean(clf[d_id], axis=(0,2))
    print(clf_temp_mean.shape)
    
    ax[d_id].imshow(clf_temp_mean, vmin=0.05, vmax=1.)
    ax[d_id].set_title(drift_type)    
    ax[d_id].set_xticks(np.arange(len(base_clfs)),base_clfs)
    ax[d_id].set_yticks(np.arange(len(n_features)),n_features)
    
    for _a, __a in enumerate(n_features):
        for _b, __b in enumerate(base_clfs):
            ax[d_id].text(_b, _a, "%.3f" % (clf_temp_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
        
        
plt.tight_layout()
plt.savefig('foo.png')
    