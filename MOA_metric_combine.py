"""
Combine metafeatures from promising groups
"""

import numpy as np
import os

np.random.seed(1233)

measures = ["clustering",
        "complexity",
        # "concept",
        # "general",
        "info-theory",
        # "itemset",
        "landmarking",
        # "model-based",
        "statistical"
        ]

streams = os.listdir('data/moa')
streams.remove('.DS_Store')
print(streams)

# MOA
for f_id in range(len(streams)):
    all = []
    for m_id, m in enumerate(measures):
        res = np.load('results/moa_%s_%s.npy' % (m, f_id))
        for i in range(res.shape[-1]-1):
            all.append(res[:,i])
            
    all.append(res[:,res.shape[-1]-1])
    all = np.array(all)
    print(all.shape)
    np.save('results/combined_moa_%i.npy' % f_id, all)
