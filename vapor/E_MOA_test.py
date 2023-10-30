import numpy as np
import matplotlib.pyplot as plt

from utils import find_real_drift
from scipy.signal import medfilt
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from strlearn.streams import ARFFParser

# stream = np.load('data/moa/stream.npy')
chunk_size = 200

# stream = ARFFParser(path='data/moa/sd_s_hyp_r2_s_hyp_r3.arff', chunk_size=chunk_size)
# stream = ARFFParser(path='data/moa/s_hyp_r1.arff', chunk_size=chunk_size)
# stream = ARFFParser(path='data/moa/s_sea_r2.arff', chunk_size=chunk_size)
stream = ARFFParser(path='data/moa/sd_s_sea_r1_s_sea_r2.arff', chunk_size=chunk_size)
n_chunks = 5000

n_drifts=4
drfs = find_real_drift(n_chunks,n_drifts)
print(drfs)

r = []
str_mean =[]
clf = MLPClassifier()
# clf = GaussianNB()

for c in range(n_chunks):
    # d = stream[c*chunk_size : (c+1)*chunk_size]

    # X, y = d[:,:-1], d[:,-1]
    X, y = stream.get_chunk()
    if c==0:
        #train
        clf.partial_fit(X, y, [0,1])
    else:
        #Test
        pred = clf.predict(X)
        r.append(balanced_accuracy_score(y, pred))
        
        #Train
        clf.partial_fit(X, y, [0,1])
    
    str_mean.append(np.mean(X, axis=0))
    
str_mean = np.array(str_mean)
print(str_mean.shape)

### plot

fig, ax = plt.subplots(4,1,figsize=(12,12))
ax[0].plot(medfilt(str_mean[:,0], 35))
ax[0].set_title('feature 1')

ax[1].plot(medfilt(str_mean[:,1], 35))
ax[1].set_title('feature 2')

ax[2].plot(medfilt(str_mean[:,2], 35))
ax[2].set_title('feature 3')

ax[3].plot(medfilt(r, 5))
ax[3].set_title('BAC')


for aa in ax:
    aa.set_xticks(drfs, drfs, rotation=90)
    aa.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')
