import utils 
import numpy as np
import matplotlib.pyplot as plt
import time

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names

res = np.load('res_clf_cls/combined_syn.npy')
print(res.shape) # features+label, drifts, reps, chunks

X = res[indexes]
print(X.shape)# f, drifts, reps, chunks

drifts = ['Sudden', 'Gradual', 'Incremental']

fig, axx = plt.subplots(3,2, figsize=(8,10), sharex=True, sharey=True)
axx[0,0].set_title('MEAN ALL')
axx[0,1].set_title('STD')


# cov entire dataset

for drift in range(3):
    covs = []

    for rep in range(5):
        
        X_all = X[:,drift,rep]
        
        for a in range(len(labels)):
            X_all[a] -= np.mean(X_all[a])
            X_all[a] /= np.std(X_all[a])

        c = np.abs(np.cov(X_all))
        covs.append(c)
    
    covs = np.mean(np.array(covs),axis=0)
    ax = axx[drift,0]
    ax.set_ylabel('%s' % (drifts[drift]))
    ax.imshow(c)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)


# calculate for metachunk
 
window = 25

for drift in range(3):
    
    collected=[]
       
    for i in range(int(X.shape[-1]/window)):
        X_w = X[:,:,:,i*window:(i+1)*window]
        
        covs = []
        for rep in range(5):
            
            X_temp = X_w[:,drift,rep]
            
            for a in range(len(labels)):
                X_temp[a] -= np.mean(X_temp[a])
                X_temp[a] /= np.std(X_temp[a])

            c = np.abs(np.cov(X_temp))
            covs.append(c)
        
        covs = np.mean(np.array(covs),axis=0)
        collected.append(covs)
    
    collected = np.array(collected)
    collected_std = np.std(collected, axis=0)
    ax = axx[drift,1]
    ax.imshow(collected_std)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)


plt.tight_layout()
plt.savefig('fig_clf/cov_syn.png')
plt.savefig('foo.png')
