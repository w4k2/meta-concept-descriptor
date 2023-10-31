"""
E2 - classification for MOA streams 
"""

import numpy as np
from sklearn import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
import os

np.random.seed(1233)

measures = ["clustering",
        "complexity",
        "concept",
        "general",
        "info-theory",
        "itemset",
        "landmarking",
        "model-based",
        "statistical"
        ]

streams = os.listdir('data/moa')
streams.remove('.DS_Store')
print(streams)

base_clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(random_state=11313),
    DecisionTreeClassifier(random_state=11313),
    MLPClassifier(random_state=11313)
]

origial_datasets=len(streams)

n_splits=2
n_repeats=5

clf_res = np.zeros((len(measures), origial_datasets, n_splits*n_repeats, len(base_clfs)))
pbar = tqdm(total=len(measures)*origial_datasets*n_splits*n_repeats*len(base_clfs))

for f_id in range(len(streams)):
    for m_id, m in enumerate(measures):
        res = np.load('results/moa_%s_%i.npy' % (m, f_id))
        
        p = np.random.permutation(res.shape[0])
        res = res[p]
    
        # print(res_rep.shape) # chunks, measures + label
        X = res[:,:-1]
        y = res[:,-1]
        
        X[np.isnan(X)]=1
        X[np.isinf(X)]=1
        
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3242)
        for fold, (train, test) in enumerate(rskf.split(X, y)):
        
            for base_id, base_c in enumerate(base_clfs):
                clf = clone(base_c)
                
                pred = clf.fit(X[train], y[train]).predict(X[test])
                acc = balanced_accuracy_score(y[test], pred)
                
                clf_res[m_id, f_id, fold, base_id] = acc
                pbar.update(1)
            
        print(m, np.mean(clf_res[m_id, f_id], axis=0))
    
np.save('results/moa_clf.npy', clf_res)
