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
fig, ax = plt.subplots(1,6,figsize=(15,15), sharex=True, sharey=True)
ax = ax.ravel()

for dataset_id, dataset in enumerate(real_streams):
    
    clf = np.load('res_clf_cls/clf_sel_real_%i.npy' % dataset_id)
    clf_mean = np.mean(clf, axis=1)
    # anova = np.load('res_clf_cls/anova_sel_real_%i.npy' % dataset_id)
    
    print(clf_mean.shape) # drfs, datasets, features, folds, clfs
    # print(anova.shape) # drfs, datasets, features, (stat, val)
    # exit()

    
    ax[dataset_id].imshow(clf_mean, vmin=0.05, vmax=1.)
    ax[dataset_id].set_title(dataset)    
    ax[dataset_id].set_xticks(np.arange(len(base_clfs)),base_clfs)
    ax[dataset_id].set_yticks(np.arange(len(n_features)),n_features)
    
    for _a, __a in enumerate(n_features):
        for _b, __b in enumerate(base_clfs):
            ax[dataset_id].text(_b, _a, "%.2f" % (clf_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
        
        
plt.tight_layout()
plt.savefig('foo.png')
    
plt.clf()

# ANOVA

anovas = []
for dataset_id, dataset in enumerate(real_streams):
    
    anova = np.load('res_clf_cls/anova_sel_real_%i.npy' % dataset_id)
    anovas.append(anova[:,0])


anovas = np.array(anovas)
anova_sum = np.nansum(anovas, axis=0)
sort_order = np.flip(np.argsort(anova_sum))

labels_measures = utils.measure_labels_selected
labels_counts = [len(l) for l in labels_measures]
labels_ids = [[c_id for _ in range(cnt)] for c_id,cnt in enumerate(labels_counts)]
labels_ids = np.array(sum(labels_ids, []))[sort_order]

labels_measures = np.array(sum(labels_measures, []))

cols=np.array(['r','g','b','gold','purple'])


fig, ax = plt.subplots(1,1,figsize=(15,10), sharex=True, sharey=True)

for dataset_id, dataset in enumerate(real_streams): 
    temp = anovas[dataset_id]
    l = labels_measures[sort_order]
    t = temp[sort_order]
    ax.bar(range(len(l)), t, alpha=0.2,color=cols[labels_ids])
    
ax.set_xticks(range(len(l)),l,rotation=90)
ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

custom_lines = [Line2D([0], [0], color=cols[0], lw=4),
                Line2D([0], [0], color=cols[1], lw=4),
                Line2D([0], [0], color=cols[2], lw=4),
                Line2D([0], [0], color=cols[3], lw=4),
                Line2D([0], [0], color=cols[4], lw=4)]
ax.legend(custom_lines, ['Clustering', 'Complexity', 'Info theory', 'Landmarking', 'Statistical'])
        
plt.tight_layout()
plt.savefig('foo.png')