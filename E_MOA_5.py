"""
E5 - experiment and presentation -- MOA streams
"""
import utils 
import numpy as np
import matplotlib.pyplot as plt

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names

res_all =[]
for i in range(12):
    res = np.load('results/combined_moa_%i.npy' % i)
    res_all.append(res)

res = np.array(res_all).reshape(4,3,119,500)
print(res.shape) # types, reps, features+label, chunks

X = res[:,:,indexes]
print(X.shape)# types, reps, selected features, chunks

types = ['RBF', 'LED', 'HYPERPLANE', 'SEA']

fig, axx = plt.subplots(2,4, figsize=(12,7), sharex=True, sharey=True)
axx[0,0].set_ylabel('MEAN ALL')
axx[1,0].set_ylabel('STD')


# cov entire dataset

for t in range(4):
    covs = []

    for rep in range(3):
        
        X_all = X[t, rep]
        
        for a in range(len(labels)):
            X_all[a] -= np.mean(X_all[a])
            X_all[a] /= np.std(X_all[a])

        c = np.abs(np.cov(X_all))
        covs.append(c)
    
    covs = np.mean(np.array(covs),axis=0)
    ax = axx[0,t]
    ax.set_title('%s' % (types[t]))
    print(np.nanmin(c), np.nanmax(c))

    im = ax.imshow(c, cmap='Greys', vmin=0, vmax=1)
    # print(np.nanmin(c), np.nanmax(c))
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)

cax_2 = axx[0,-1].inset_axes([1.04, 0.0, 0.05, 1.0])
fig.colorbar(im, ax=axx[0,0], cax=cax_2)  

# calculate for metachunk
 
window = 25

for t in range(4):
    
    collected=[]
       
    for i in range(int(X.shape[-1]/window)):
        X_w = X[:,:,:,i*window:(i+1)*window]
        
        covs = []
        for rep in range(3):
            
            X_temp = X_w[t, rep]
            
            for a in range(len(labels)):
                X_temp[a] -= np.mean(X_temp[a])
                X_temp[a] /= np.std(X_temp[a])

            c = np.abs(np.cov(X_temp))
            covs.append(c)
        
        covs = np.mean(np.array(covs),axis=0)
        collected.append(covs)
    
    collected = np.array(collected)
    collected_std = np.std(collected, axis=0)
    ax = axx[1,t]
    im = ax.imshow(collected_std, cmap='Greys', vmin=0, vmax=0.15)
    ax.set_xlim(collected_std.shape[0]-.5,-.5)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    
cax_2 = axx[1,-1].inset_axes([1.04, 0.0, 0.05, 1.0])
fig.colorbar(im, ax=axx[0,0], cax=cax_2)


plt.tight_layout()
plt.savefig('figures/fig_clf/cov_moa.png')
plt.savefig('figures/fig_clf/cov_moa.eps')
plt.savefig('foo.png')