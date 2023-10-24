"""
E1 - scatterplot - rzeczywiste
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
import utils
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

cmap = matplotlib.cm.get_cmap('rainbow')

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
        res = np.load('results/real_%s_%s.npy' % (f_id, m))
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
        
        # Shuffle order and establish ranges for grid
        shuffler = np.array(list(range(X.shape[0])))
        np.random.shuffle(shuffler)
        
        _X = X - np.min(X, axis=0)
        _X = _X / np.max(_X, axis=0)

        labels = np.unique(y)
        colors = cmap(np.linspace(0,1,len(labels)))

        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                aa = ax[i,j]
                if j > i:
                    aa.cla()
                    aa.set_yticks([])
                    aa.set_xticks([])
                    aa.spines['top'].set_visible(False)
                    aa.spines['right'].set_visible(False)
                    aa.spines['left'].set_visible(False)
                    aa.spines['bottom'].set_visible(False)

                else:    
                    ax[i,j].cla()
                    ax[i,j].set_yticks([])
                    ax[i,j].set_xticks([])
                    
                    aa.hlines(np.linspace(0,1,5)[1:-1], 0, 1, lw=.25, ls=':', color='black')
                    aa.vlines(np.linspace(0,1,5)[1:-1], 0, 1, lw=.25, ls=':', color='black')
                    
                    if i != j:
                        aa.scatter(_X[shuffler,i], _X[shuffler,j], c=y[shuffler], 
                                linewidth=0, alpha=1, s=2, edgecolors=None, cmap='rainbow')
                        aa.set_xlim(-.1,1.1)
                        aa.set_ylim(-.1,1.1)
                    else:
                        for lidx, label in enumerate(labels):
                            print('label', label)
                            aa.hist(_X[y==label,i], bins = 32, color=colors[lidx],
                                    range=(0,1),
                                    alpha=.5)

                    aa.grid(ls=':')
                    aa.spines['top'].set_visible(False)
                    aa.spines['right'].set_visible(False)

                    if j==0:
                        ax[i,j].set_ylabel(names[i])
                    if i==X.shape[1]-1:
                        ax[i,j].set_xlabel(names[j])
        
                    
        aa = plt.subplot(448, projection='polar')
                
        yy = np.unique(y, return_counts=True)
        print(yy)
        aa.scatter(yy[0]/(len(yy[0]))*np.pi*2, 
                yy[1],
                c=cmap(yy[0]/len(labels)),
                linewidth=0, alpha=1, s=15, edgecolors=None)
        
        for a,b in zip(*yy):
            print(a,b)
            xa = (a/(len(yy[0])))*np.pi*2
            aa.plot([xa, xa], [0,b], c=cmap(a/len(labels)), lw=1)
        
        aa.set_ylim(0, np.max(yy[1])*1.5)
        
        
        aa.set_yticks([])
        aa.set_xticks((yy[0]/(len(labels)))*np.pi*2, ['' for _ in yy[0]])
        aa.grid(ls=':')

        aa = plt.subplot(443)     
        
        _X[np.isnan(_X)] = 1        
        pca_X = PCA(n_components=2).fit_transform(_X)
        pca_X -= np.mean(pca_X, axis=0)
        pca_X /= np.std(pca_X, axis=0)
        
        aa.scatter(*pca_X.T, c=y, cmap='rainbow',
                linewidth=0, alpha=1, s=2, edgecolors=None)
        
        aa.set_yticks([])
        aa.set_xticks([])

        #aa.hlines(np.linspace(0,1,5)[1:-1], 0, 1, lw=.25, ls=':', color='black')
        #aa.vlines(np.linspace(0,1,5)[1:-1], 0, 1, lw=.25, ls=':', color='black')
        aa.set_title('PCA')
        
        aa = plt.subplot(444)             
        tsne_X = TSNE(n_components=2, n_iter=400, n_iter_without_progress=100, verbose=True).fit_transform(_X)
        #tsne_X = TSNE(n_components=2, n_iter=250, n_iter_without_progress=50, verbose=True).fit_transform(_X)
        tsne_X -= np.mean(tsne_X, axis=0)
        tsne_X /= np.std(tsne_X, axis=0)
        
        aa.scatter(*tsne_X.T, c=y, cmap='rainbow',
                linewidth=0, alpha=1, s=2, edgecolors=None)
        
        aa.set_yticks([])
        aa.set_xticks([])

        #aa.hlines(np.linspace(0,1,5)[1:-1], 0, 1, lw=.25, ls=':', color='black')
        #aa.vlines(np.linspace(0,1,5)[1:-1], 0, 1, lw=.25, ls=':', color='black')
        aa.set_title('t-SNE')
                                        
        plt.tight_layout()
        plt.savefig('figures/fig_rel/%s_%i.png' % (m, f_id))
        plt.savefig('figures/fig_rel/%s_%i.eps' % (m, f_id))
        plt.savefig('foo.png')
        #exit()
        