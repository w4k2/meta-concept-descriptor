import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier

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

for m in measures:
    res = np.load('res/semi_%s.npy' % m)
    print(res.shape) # datasets, drifts, reps, chunks, measures + label
    for dataset_id, dataset in enumerate(static_data):
        for r in range(1):
            
            res_iter = res[dataset_id,:,r]
            perm = np.random.permutation(res_iter.shape[1])
            res_iter = res_iter[:,perm]
            
            for drf_id, drf in enumerate(['nearest', 'cubic']):
                X, y = res_iter[drf_id,:,:-1], res_iter[drf_id,:,-1]
                    
                #EH
                X[np.isnan(X)]=1
                
                if  X.shape[1]>8:
                    # Feature Selection
                    skb = SelectKBest(k=8)
                    X = skb.fit_transform(X, y)
                                        
                fig, ax = plt.subplots(X.shape[1],X.shape[1],figsize=(12,12))
                plt.suptitle('%s %s %s rep:%i' % (m, drf, static_data[dataset_id], r))
                for i in range(X.shape[1]):
                    for j in range(X.shape[1]):
                        ax[i,j].cla()
                        ax[i,j].set_yticks([])
                        ax[i,j].set_xticks([])
                        ax[i,j].scatter(X[:,i], X[:,j], c=y, alpha=0.005, s=50, edgecolors=None, cmap='Set1')
                    
                plt.tight_layout()
                plt.savefig('fig_semi/%s_%s_%s_%i.png' % (m, drf, static_data[dataset_id], r))
                plt.savefig('foo.png')
                