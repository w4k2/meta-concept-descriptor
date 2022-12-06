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
fig, ax = plt.subplots(2,6,figsize=(10,15), sharex=True, sharey=True)

for d_id, drift_type in enumerate(['Nearest', 'Cubic']):   
    for dataset_id, dataset in enumerate(static_data):
 
        clf_temp = clf[dataset_id, d_id]
        print(clf_temp.shape)
        clf_temp_mean = np.mean(clf_temp, axis=1)
        print(clf_temp_mean.shape)
        
        ax[d_id,dataset_id].imshow(clf_temp_mean, vmin=0.05, vmax=1.)
        ax[d_id,dataset_id].set_title('%s %s' % (dataset, drift_type[0]))
        ax[d_id,dataset_id].set_xticks(np.arange(len(base_clfs)),base_clfs, rotation=90)
        ax[d_id,dataset_id].set_yticks(np.arange(len(n_features)),n_features)
        
        # for _a, __a in enumerate(n_features):
        #     for _b, __b in enumerate(base_clfs):
        #         ax[d_id,dataset_id].text(_b, _a, "%.3f" % (clf_temp_mean[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
            
        
plt.tight_layout()
plt.savefig('fig_clf/sel_semi.png')    
plt.clf()

# ANOVA

anova_sum = np.nansum(anova[:,:,:,0], axis=(0,1))
sort_order = np.flip(np.argsort(anova_sum))

labels_measures = utils.measure_labels_selected
labels_counts = [len(l) for l in labels_measures]
labels_ids = [[c_id for _ in range(cnt)] for c_id,cnt in enumerate(labels_counts)]
labels_ids = np.array(sum(labels_ids, []))[sort_order]

labels_measures = np.array(sum(labels_measures, []))

cols=np.array(['r','g','b','gold','purple'])


fig, ax = plt.subplots(2,1,figsize=(15,10), sharex=True, sharey=True)

for d_id, drift_type in enumerate(['Nearest', 'Cubic']):   
    for dataset_id, dataset in enumerate(static_data): 
        ax[d_id].set_title(drift_type)    
        temp = anova[dataset_id,d_id,:,0]
        l = labels_measures[sort_order]
        t = temp[sort_order]
        ax[d_id].bar(range(len(l)), t, alpha=0.2,color=cols[labels_ids])
    ax[d_id].set_xticks(range(len(l)),l,rotation=90)
    ax[d_id].grid()
    ax[d_id].spines['top'].set_visible(False)
    ax[d_id].spines['right'].set_visible(False)

custom_lines = [Line2D([0], [0], color=cols[0], lw=4),
                Line2D([0], [0], color=cols[1], lw=4),
                Line2D([0], [0], color=cols[2], lw=4),
                Line2D([0], [0], color=cols[3], lw=4),
                Line2D([0], [0], color=cols[4], lw=4)]
ax[0].legend(custom_lines, ['Clustering', 'Complexity', 'Info theory', 'Landmarking', 'Statistical'])
        
plt.tight_layout()
plt.savefig('fig_clf/anova_semi.png')



# REDUCED

reduced = np.load('res_clf_cls/semi_clf_reduced.npy')
print(reduced.shape) # 6, 2, 10, 5
# exit()

reduced_mean = np.mean(reduced, axis=2)

fig, ax = plt.subplots(6, 2, figsize=(8,15), sharex=True, sharey=True)

for dataset_id, dataset in enumerate(static_data):
    for drf_id, drift_type in enumerate(['nearest', 'cubic']):    
        img = np.zeros((2,5))
        img[0] = reduced_mean[dataset_id,drf_id]
        img[1] = np.mean(clf[dataset_id, drf_id, -1], axis=0)
        
        ax[dataset_id,drf_id].imshow(img, vmin=0.05, vmax=1)
        ax[dataset_id,drf_id].set_title('%s %s' % (dataset, drift_type))
        
        ax[dataset_id,drf_id].set_xticks(range(len(base_clfs)), base_clfs)
        ax[dataset_id,drf_id].set_yticks(range(2), ['reduced', 'full'])
        
        for _a, __a in enumerate(['reduced', 'full']):
            for _b, __b in enumerate(base_clfs):
                ax[dataset_id,drf_id].text(_b, _a, "%.3f" % (img[_a, _b]) , va='center', ha='center', c='white', fontsize=11)
        

plt.tight_layout()
plt.savefig('fig_clf/reduced_semi.png')
plt.savefig('foo.png')
