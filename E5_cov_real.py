"""
Obliczenia + Plot.

E5 - real
"""
import utils 
import numpy as np
import matplotlib.pyplot as plt

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names


real_streams = [
    'covtypeNorm-1-2vsAll',
    'electricity',
    'poker-lsn-1-2vsAll',
    'INSECTS-abrupt',
    'INSECTS-gradual',
    'INSECTS-incremental'
    ]

fig, axx = plt.subplots(2,6, figsize=(20,7.5), sharex=True, sharey=True)

axx[0,0].set_ylabel('ALL')
axx[1,0].set_ylabel('STD')

for f_id, f in enumerate(real_streams):
    
    res = np.load('results/combined_real_%i.npy' % f_id)
    X = res[indexes]
    
    # cov entire dataset
    X_all = np.copy(X)
    for a in range(len(labels)):
        X_all[a] -= np.mean(X_all[a])
        X_all[a] /= np.std(X_all[a])

    c = np.abs(np.cov(X_all))

    ax = axx[0, f_id]
    ax.set_title('%s' % (f))
    # print(np.nanmin(c), np.nanmax(c))
    im = ax.imshow(c, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    
    cax_2 = axx[0,-1].inset_axes([1.04, 0.0, 0.05, 1.0])
    fig.colorbar(im, ax=axx[0,0], cax=cax_2)  

    # calculate for metachunk
    collected=[]
    window = 25

    for i in range(int(X.shape[-1]/window)):
        
        # print(i*window,(i+1)*window)
        X_w = X[:,i*window:(i+1)*window]
        # print(X_w.shape)
        
        for a in range(len(labels)):
            m = np.mean(X_w[a])
            X_w[a] -= m
            s = np.std(X_w[a])
            X_w[a] /= s
        

        c = np.abs(np.cov(X_w))
        collected.append(c)
    
    std_collected = np.std(np.array(collected), axis=0)
    # std_collected = np.mean(np.array(collected), axis=0)
    ax = axx[1,f_id]
    im = ax.imshow(std_collected, cmap='Greys',vmin=0,vmax=0.4)
    ax.set_xlim(std_collected.shape[0]-.5,-.5)

    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    
    cax_2 = axx[1,-1].inset_axes([1.04, 0.0, 0.05, 1.0])
    fig.colorbar(im, ax=axx[0,0], cax=cax_2)  
       
plt.tight_layout()
plt.savefig('figures/fig_clf/cov_real.png')
plt.savefig('figures/fig_clf/cov_real.eps')
