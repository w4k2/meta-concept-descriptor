import matplotlib.pyplot as plt
import numpy as np
import utils
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

real_streams_full = [
    'real_streams/covtypeNorm-1-2vsAll-pruned.arff',
    'real_streams/electricity.npy',
    'real_streams/poker-lsn-1-2vsAll-pruned.arff',
    'real_streams/INSECTS-abrupt_imbalanced_norm.arff',
    'real_streams/INSECTS-gradual_imbalanced_norm.arff',
    'real_streams/INSECTS-incremental_imbalanced_norm.arff'
    ]

limit=5

for f_id in range(len(real_streams)):
    for m_id, m in enumerate(measures):
        res = np.load('res/real_%s_%s.npy' % (f_id, m))
        if res.shape[0]==0:
            continue
        # print(f_id, m)
        # print(res.shape) # drfs, reps, chunks, measures + label
        
        X = res[:,:-1]
        y=[]
        
        concept=0
        fname=(real_streams_full[f_id].split('/')[1]).split('.')[0]
        drfs = np.load('real_streams_gt/%s.npy' % fname)
        for i in range(X.shape[0]):
            if i in drfs:
                concept+=1
            y.append(concept)
            
        perm = np.random.permutation(res.shape[0])
        X = X[perm]
        y = np.array(y)[perm]
          
        #EH
        X[np.isnan(X)]=1
        X[np.isinf(X)]=1
        names = [n[:6] for n in utils.measure_labels[m_id]]
        
        if  X.shape[1]>limit:
            # Feature Selection
            pca = PCA(n_components=int(np.rint(np.sqrt(X.shape[1]))))
            pca.fit(X)
            av = np.sum(np.abs(pca.components_), axis=0)
            av_s=np.flip(np.argsort(av))[:limit]
            
            X = X[:,av_s]     
            names = np.array(names)[av_s]      

        fig, ax = plt.subplots(X.shape[1],X.shape[1],figsize=(7,7))
        plt.suptitle('%s %s' % (m, real_streams[f_id]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                ax[i,j].cla()
                ax[i,j].set_yticks([])
                ax[i,j].set_xticks([])
                ax[i,j].scatter(X[:,i], X[:,j], c=y, linewidth=0, alpha=0.2, s=20, edgecolors=None, cmap='rainbow')
                if j==0:
                    ax[i,j].set_ylabel(names[i])
                if i==X.shape[1]-1:
                    ax[i,j].set_xlabel(names[j])                    
        plt.tight_layout()
        plt.savefig('fig_rel/%s_%i.png' % (m, f_id))
        plt.savefig('foo.png')
        