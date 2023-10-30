import numpy as np
import matplotlib.pyplot as plt
from utils import find_real_drift
from strlearn.streams import ARFFParser

chunk_size = 200
n_chunks = 5000

# stream = ARFFParser(path='data/moa/sd_s_hyp_r2_s_hyp_r3.arff', chunk_size=chunk_size)
# stream = ARFFParser(path='data/moa/s_hyp_r1.arff', chunk_size=chunk_size)
stream = ARFFParser(path='data/moa/s_sea_r2.arff', chunk_size=chunk_size)

for c in range(1000):
    try:
        X, y = stream.get_chunk()
    except:
        break
    
    fig, ax = plt.subplots(3,3,figsize=(7,7))
    plt.suptitle('c:%i | %i-%i' % (c, c*chunk_size , (c+1)*chunk_size))
    
    for f_id1 in range(3):
        for f_id2 in range(3):
            ax[f_id1, f_id2].scatter(X[:, f_id1], X[:, f_id2], c=y, s=2)
            
            # ax[f_id1, f_id2].set_xlim(-5,15)
            # ax[f_id1, f_id2].set_ylim(-5,15)

    plt.tight_layout()
    plt.savefig('temp/%04d.png' % c)
    plt.savefig('foo.png')
