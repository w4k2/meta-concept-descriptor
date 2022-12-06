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

fig, axx = plt.subplots(2,3, figsize=(23,15), sharex=True, sharey=True)
axx = axx.ravel()

for f_id, f in enumerate(real_streams):
    res = np.load('res_clf_cls/combined_real_%i.npy' % f_id)
    print(res.shape) # features+label, drifts, reps, chunks

    X = res[indexes]
    print(X.shape)

    # normalize
    for i in range(16):
        X[i] -= np.mean(X[i])    
        X[i] /= np.std(X[i]) + 0.00001

    c = np.abs(np.cov(X))

    ax = axx[f_id]
    ax.set_title('%s' % (f))
    ax.imshow(c)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
                
plt.tight_layout()
plt.savefig('foo.png')
