"""
Combine metafeatures from promising groups
"""

import numpy as np

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

real_streams_full = [
    'real_streams/covtypeNorm-1-2vsAll-pruned.arff',
    'real_streams/electricity.npy',
    'real_streams/poker-lsn-1-2vsAll-pruned.arff',
    'real_streams/INSECTS-abrupt_imbalanced_norm.arff',
    'real_streams/INSECTS-gradual_imbalanced_norm.arff',
    'real_streams/INSECTS-incremental_imbalanced_norm.arff'
    ]

#SYNTHETIC
all = []

for m_id, m in enumerate(measures):
    res = np.load('res/%s.npy' % m)
    for i in range(res.shape[-1]-1):
        all.append(res[:,:,:,i])

all.append(res[:,:,:,res.shape[-1]-1])
all = np.array(all)
print(all.shape)
np.save('res_clf_cls/combined_syn.npy', all)


#SEMI
all = []

for m_id, m in enumerate(measures):
    res = np.load('res/semi_%s.npy' % m)
    for i in range(res.shape[-1]-1):
        all.append(res[:,:,0,:,i])
        
all.append(res[:,:,0,:,res.shape[-1]-1])
all = np.array(all)
print(all.shape)
np.save('res_clf_cls/combined_semi.npy', all)


# #REAL
for f_id in range(len(real_streams_full)):
    all = []
    for m_id, m in enumerate(measures):
        res = np.load('res/real_%s_%s.npy' % (f_id, m))
        for i in range(res.shape[-1]-1):
            all.append(res[:,i])
            
    all.append(res[:,res.shape[-1]-1])
    all = np.array(all)
    print(all.shape)
    np.save('res_clf_cls/combined_real_%i.npy' % f_id, all)
