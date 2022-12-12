"""
Obliczenia + Plot.

E5 - semi
"""
import utils 
import numpy as np
import matplotlib.pyplot as plt
import time

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names
streams = ['australian',
        'banknote',
        'diabetes',
        'german',
        'vowel0',
        'wisconsin'
        ]


fig, axx = plt.subplots(6,2, figsize=(8,20), sharex=True, sharey=True)
axx[0,0].set_title('MEAN ALL')
axx[0,1].set_title('STD')

res = np.load('res_clf_cls/combined_semi.npy')
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
    ax = axx[rep,0]
    ax.set_ylabel('%s' % (streams[rep]))
    ax.imshow(c, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)


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
    ax = axx[rep,1]
    ax.imshow(collected_std, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)


plt.tight_layout()
plt.savefig('fig_clf/cov_semi.png')
plt.savefig('foo.png')
time.sleep(0.5)
