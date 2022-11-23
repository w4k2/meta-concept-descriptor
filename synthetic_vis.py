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

for m in measures:
    res = np.load('res/%s.npy' % m)
    print(res.shape) # drfs, reps, chunks, measures + label
    
    for r in range(1):
    # for r in range(res.shape[1]):
        
        res_iter = res[:,r]
        perm = np.random.permutation(res_iter.shape[1])
        res_iter = res_iter[:,perm]
        
        for drf_id, drf in enumerate(['sudd', 'inc', 'grad']):
            X, y = res_iter[drf_id,:,:-1], res_iter[drf_id,:,-1]
                
            #EH
            X[np.isnan(X)]=1
            
            if  X.shape[1]>8:
                # Feature Selection
                skb = SelectKBest(k=8)
                X = skb.fit_transform(X, y)
                                    
            fig, ax = plt.subplots(X.shape[1],X.shape[1],figsize=(12,12))
            plt.suptitle('%s %s rep:%i' % (m, drf, r))
            for i in range(X.shape[1]):
                for j in range(X.shape[1]):
                    ax[i,j].cla()
                    ax[i,j].set_yticks([])
                    ax[i,j].set_xticks([])
                    ax[i,j].scatter(X[:,i], X[:,j],c=y, alpha=0.05, s=1)
                
            plt.tight_layout()
            plt.savefig('fig_syn/%s_%s_%i.png' % (m, drf, r))
            plt.savefig('foo.png')
            