import utils 
import numpy as np
import matplotlib.pyplot as plt
import time

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names

res = np.load('res_clf_cls/combined_semi.npy')
print(res.shape) # features+label, drifts, reps, chunks

X = res[indexes]
print(X.shape)

streams = ['australian',
        'banknote',
        'diabetes',
        'german',
        'vowel0',
        'wisconsin'
        ]

window = 25

for i in range(int(X.shape[-1]/window)):
    
    print(i*window,(i+1)*window)
    X_w = X[:,:,:,i*window:(i+1)*window]
    print(X_w.shape)
            
    covs = np.zeros((2,6,16,16))
        
    for drift in range(2):
        for rep in range(6):
            temp = X_w[:,rep,drift]
            for f in range(16):
                m = np.mean(temp[f])
                temp[f] -= m
                s = np.std(temp[f])
                temp[f] /= s
            covs[drift, rep] = np.abs(np.cov(temp))

    covs_mean = np.mean(covs, axis=0)

    fig, axx = plt.subplots(2,3, figsize=(22,12), sharex=True, sharey=True)
    axx = axx.ravel()

    for stream in range(6):
        ax = axx[stream]
        ax.set_title('%s' % (streams[stream]))
        ax.imshow(covs_mean[stream])
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
                
    plt.tight_layout()
    # plt.savefig('fig_clf/cov_semi.png')
    plt.savefig('foo.png')
    time.sleep(0.5)
