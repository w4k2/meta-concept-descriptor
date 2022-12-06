import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest

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
limit=5

for m_id, m in enumerate(measures):
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
                names = [n[:6] for n in utils.measure_labels[m_id]]

                if  X.shape[1]>limit:
                    # Feature Selection
                    skb = SelectKBest(k=limit)
                    X = skb.fit_transform(X, y)
                    names = skb.get_feature_names_out(input_features=names)
                    
                                        
                fig, ax = plt.subplots(X.shape[1],X.shape[1],figsize=(7,7))
                plt.suptitle('%s %s %s rep:%i' % (m, drf, static_data[dataset_id], r))
                for i in range(X.shape[1]):
                    for j in range(X.shape[1]):
                        ax[i,j].cla()
                        ax[i,j].set_yticks([])
                        ax[i,j].set_xticks([])
                        ax[i,j].scatter(X[:,i], X[:,j], c=y, linewidth=0, alpha=0.05, s=5, edgecolors=None, cmap='rainbow')
                        if j==0:
                            ax[i,j].set_ylabel(names[i])
                        if i==X.shape[1]-1:
                            ax[i,j].set_xlabel(names[j])                    
                plt.tight_layout()
                plt.savefig('fig_semi/%s_%s_%s_%i.png' % (m, drf, static_data[dataset_id], r))
                plt.savefig('foo.png')
