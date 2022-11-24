import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

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

for f_id in range(len(real_streams)):
    for m in measures:
        res = np.load('res/real_%s_%s.npy' % (f_id, m))
        if res.shape[0]==0:
            print(f_id)
            continue
        # print(f_id, m)
        print(res.shape) # drfs, reps, chunks, measures + label
        perm = np.random.permutation(res.shape[1])
        res = res[:,perm]
        
        X = res
            
        #EH
        X[np.isnan(X)]=1
        
        if  X.shape[1]>8:
            # Feature Selection
            pca = PCA(n_components=int(np.rint(np.sqrt(X.shape[1]))))
            pca.fit(X)
            av = np.sum(np.abs(pca.components_), axis=0)
            av_s=np.flip(np.argsort(av))[:8]
            
            X = X[:,av_s]           

        fig, ax = plt.subplots(X.shape[1],X.shape[1],figsize=(12,12))
        plt.suptitle('%s %s' % (m, real_streams[f_id]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                ax[i,j].cla()
                ax[i,j].set_yticks([])
                ax[i,j].set_xticks([])
                ax[i,j].scatter(X[:,i], X[:,j], alpha=0.1, s=10)
            
        plt.tight_layout()
        plt.savefig('fig_rel/%s_%i.png' % (m, f_id))
        plt.savefig('foo.png')
        