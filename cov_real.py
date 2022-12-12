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

fig, axx = plt.subplots(6,2, figsize=(8,20), sharex=True, sharey=True)

axx[0,0].set_title('ALL')
axx[0,1].set_title('STD')

for f_id, f in enumerate(real_streams):
    
    res = np.load('res_clf_cls/combined_real_%i.npy' % f_id)
    print(res.shape) # features+label, drifts, reps, chunks
    
    X = res[indexes]
    print(X.shape)
    
    # cov entire dataset
    X_all = np.copy(X)
    for a in range(len(labels)):
        X_all[a] -= np.mean(X_all[a])
        X_all[a] /= np.std(X_all[a])

    c = np.abs(np.cov(X_all))

    ax = axx[f_id,0]
    ax.set_ylabel('%s' % (f))
    ax.imshow(c, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    
    # calculate for metachunk
    collected=[]
    window = 25

    for i in range(int(X.shape[-1]/window)):
        
        print(i*window,(i+1)*window)
        X_w = X[:,i*window:(i+1)*window]
        print(X_w.shape)
        
        for a in range(len(labels)):
            m = np.mean(X_w[a])
            X_w[a] -= m
            s = np.std(X_w[a])
            X_w[a] /= s
        

        c = np.abs(np.cov(X_w))
        collected.append(c)
    
    std_collected = np.std(np.array(collected), axis=0)
    # std_collected = np.mean(np.array(collected), axis=0)
    ax = axx[f_id,1]
    ax.imshow(std_collected, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
                
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_clf/cov_real.png')
# time.sleep(0.5)
