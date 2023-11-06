import os
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import ARFFParser
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.signal import medfilt

streams = os.listdir('data/moa')
streams.remove('.DS_Store')
print(streams)

chunks = 500

fig, ax = plt.subplots(4,3,figsize=(12,7), sharex=True, sharey=True)
ax = ax.ravel()

for s_id, s in enumerate(streams):
    data = ARFFParser('data/moa/%s' % s)
    clf = GaussianNB()
    scores = []

    for c in range(chunks):
        X, y = data.get_chunk()
        if c==0:
            clf.fit(X, y)
        else:
            scores.append(accuracy_score(y, clf.predict(X)))
    
    ax[s_id].plot(medfilt(scores,11))
    ax[s_id].grid(ls=':')
    ax[s_id].set_title(s.split('.')[0])
    if s_id in [0,3,6,9]:
       ax[s_id].set_ylabel('accuracy')
        
    plt.tight_layout()
    plt.savefig('foo.png')