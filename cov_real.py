import utils 
import numpy as np
import matplotlib.pyplot as plt
import time 

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
    
    window = 25

    for i in range(int(X.shape[-1]/window)):
        
        print(i*window,(i+1)*window)
        X_w = X[:,i*window:(i+1)*window]
        print(X_w.shape)
        
        for a in range(16):
            m = np.mean(X_w[a])
            X_w[a] -= m
            s = np.std(X_w[a])
            X_w[a] /= s
        

        c = np.abs(np.cov(X_w))

        ax = axx[f_id]
        ax.set_title('%s' % (f))
        ax.imshow(c)
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
                    
        plt.tight_layout()
        plt.savefig('foo.png')
        time.sleep(0.5)
