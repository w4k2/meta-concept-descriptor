import utils 
import numpy as np
import matplotlib.pyplot as plt
import time

indexes = utils.selected2_indexes
labels = utils.selected2_measure_names

res = np.load('res_clf_cls/combined_syn.npy')
print(res.shape) # features+label, drifts, reps, chunks

X = res[indexes]

window = 25

for i in range(int(X.shape[-1]/window)):
    
    print(i*window,(i+1)*window)
    X_w = X[:,:,:,i*window:(i+1)*window]
    print(X_w.shape)
    
    covs = np.zeros((3,5,16,16))

    for drift in range(3):
        for rep in range(5):
            temp = X_w[:,drift,rep]
            for f in range(16):
                # if m == None:
                    m = np.mean(temp[f])
                    temp[f] -= m
                    s = np.std(temp[f])
                    temp[f] /= s
                # else:
                    # temp[f] -= m
                    # temp[f] /= s

            covs[drift, rep] = np.abs(np.cov(temp))
    
    covs_mean = np.mean(covs, axis=1)
    print(covs_mean.shape)

    fig, ax = plt.subplots(1,3, figsize=(18,8), sharex=True, sharey=True)
    plt.suptitle('%s/%s' % (i,int(X.shape[-1]/window)))

    for d_id, drift in enumerate(['Sudden', 'Gradual', 'Incremental']):
        ax[d_id].imshow(covs_mean[d_id], vmin=0, vmax=1)
        ax[d_id].set_title(drift)
        ax[d_id].set_xticks(range(len(labels)), labels, rotation=90)
        ax[d_id].set_yticks(range(len(labels)), labels)

    plt.tight_layout()
    # plt.savefig('fig_clf/cov_syn.png')
    plt.savefig('foo.png')
    time.sleep(0.5)
