"""
E5 - experiment and presentation -- semi-synthetic streams
"""
import utils 
import numpy as np
import matplotlib.pyplot as plt

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names
streams = ['australian',
        'banknote',
        'diabetes',
        'german',
        'vowel0',
        'wisconsin'
        ]


fig, axx = plt.subplots(2,6, figsize=(20,7.5), sharex=True, sharey=True)
axx[0,0].set_ylabel('MEAN ALL')
axx[1,0].set_ylabel('STD')

res = np.load('results/combined_semi.npy')
print(res.shape) # features+label, drifts, reps, chunks

X = res[indexes]
print(X.shape)

# cov entire dataset

for rep in range(6):
    covs = []

    for drift in range(2):
        
        X_all = X[:,rep,drift]
        
        for a in range(len(labels)):
            X_all[a] -= np.mean(X_all[a])
            X_all[a] /= np.std(X_all[a])

        c = np.abs(np.cov(X_all))
        covs.append(c)
    
    covs = np.mean(np.array(covs),axis=0)
    ax = axx[0,rep]
    ax.set_title('%s' % (streams[rep]))
    print(np.nanmin(c), np.nanmax(c))

    im = ax.imshow(c, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)

cax_2 = axx[0,-1].inset_axes([1.04, 0.0, 0.05, 1.0])
fig.colorbar(im, ax=axx[0,0], cax=cax_2)  


# calculate for metachunk
 
window = 25

for rep in range(6):
    
    collected=[]
       
    for i in range(int(X.shape[-1]/window)):
        X_w = X[:,:,:,i*window:(i+1)*window]
        
        covs = []
        for drift in range(2):
            
            X_temp = X_w[:,rep,drift]
            
            for a in range(len(labels)):
                X_temp[a] -= np.mean(X_temp[a])
                X_temp[a] /= np.std(X_temp[a])

            c = np.abs(np.cov(X_temp))
            covs.append(c)
        
        covs = np.mean(np.array(covs),axis=0)
        collected.append(covs)
    
    collected = np.array(collected)
    collected_std = np.std(collected, axis=0)
    ax = axx[1,rep]
    # print(np.nanmin(collected_std), np.nanmax(collected_std))
    im = ax.imshow(collected_std, cmap='Greys', vmin=0, vmax=0.2)
    ax.set_xlim(collected_std.shape[0]-.5,-.5)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)

cax_2 = axx[1,-1].inset_axes([1.04, 0.0, 0.05, 1.0])
fig.colorbar(im, ax=axx[0,0], cax=cax_2)  

plt.tight_layout()
plt.savefig('figures/fig_clf/cov_semi.png')
plt.savefig('figures/fig_clf/cov_semi.eps')

