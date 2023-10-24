"""
E1 - scatterplot - semi
"""

import utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
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

static_data = ['australian',
               'banknote',
               'diabetes',
               'german',
               'vowel0',
               'wisconsin'
               ]
limit=5

for m_id, m in enumerate(measures):
    res = np.load('results/semi_%s.npy' % m)
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
                    
                                        
                fig, ax = plt.subplots(X.shape[1],
                                       X.shape[1],
                                       figsize=(6,6))
                
                plt.suptitle('%s %s %s' % (m, drf, static_data[dataset_id]))
                
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
                                aa.set_xlim(0,1)
                                aa.set_ylim(0,1)
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
                            
                            # ax[i,j].scatter(X[:,i], X[:,j], c=y, linewidth=0, alpha=0.05, s=5, edgecolors=None, cmap='rainbow')
                            # ax[i,j].spines['top'].set_visible(False)
                            # ax[i,j].spines['right'].set_visible(False)
                            # if j==0:
                            #     ax[i,j].set_ylabel(names[i])
                            # if i==X.shape[1]-1:
                            #     ax[i,j].set_xlabel(names[j])   
                            
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
                plt.savefig('figures/fig_semi/%s_%s_%s_%i.png' % (m, drf, static_data[dataset_id], r))
                plt.savefig('figures/fig_semi/%s_%s_%s_%i.eps' % (m, drf, static_data[dataset_id], r))
                plt.savefig('foo.png', dpi=200)
                #exit()
