import utils 
import numpy as np
import matplotlib.pyplot as plt

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names

res = np.load('res_clf_cls/combined_semi.npy')
print(res.shape) # features+label, drifts, reps, chunks

X = res[indexes]
print(X.shape)

# normalize
for i in range(16):
    X[i] -= np.mean(X[i])    
    X[i] /= np.std(X[i]) + 0.00001
    
covs = np.zeros((2,6,16,16))
    
for drift in range(2):
    for rep in range(6):
        temp = X[:,rep,drift]
        print(temp.shape)
        c = np.abs(np.cov(temp))
        covs[drift, rep] = c

covs_mean = np.mean(covs, axis=0)

fig, axx = plt.subplots(2,3, figsize=(22,12), sharex=True, sharey=True)
axx = axx.ravel()

streams = ['australian',
    'banknote',
    'diabetes',
    'german',
    'vowel0',
    'wisconsin'
    ]

for stream in range(6):
    ax = axx[stream]
    ax.set_title('%s' % (streams[stream]))
    ax.imshow(covs_mean[stream])
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
            
plt.tight_layout()
plt.savefig('foo.png')
