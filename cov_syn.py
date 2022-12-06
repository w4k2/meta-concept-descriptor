import utils 
import numpy as np
import matplotlib.pyplot as plt

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names

res = np.load('res_clf_cls/combined_syn.npy')
print(res.shape) # features+label, drifts, reps, chunks

X = res[indexes]
print(X.shape)

# normalize
for i in range(16):
    X[i] -= np.mean(X[i])    
    X[i] /= np.std(X[i]) + 0.00001
    
covs = np.zeros((3,5,16,16))
    
for drift in range(3):
    for rep in range(5):
        temp = X[:,drift,rep]
        print(temp.shape)
        c = np.abs(np.cov(temp))
        covs[drift, rep] = c

covs_mean = np.mean(covs, axis=(0,1))

fig, ax = plt.subplots(1,1, figsize=(12,12), sharex=True, sharey=True)

ax.imshow(covs_mean)
ax.set_xticks(range(len(labels)), labels, rotation=90)
ax.set_yticks(range(len(labels)), labels)

plt.tight_layout()
plt.savefig('foo.png')
