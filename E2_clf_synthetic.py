"""
E2 - classification for synthetic streams
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

base_clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(random_state=11313),
    DecisionTreeClassifier(random_state=11313),
    MLPClassifier(random_state=11313)
]

n_splits=2
n_repeats=5

n_drift_types = 3
stream_reps=5

clf_res = np.zeros((len(measures), n_drift_types, stream_reps, n_splits*n_repeats, len(base_clfs)))

pbar = tqdm(total=len(measures)*n_drift_types*stream_reps*n_splits*n_repeats*len(base_clfs))

for m_id, m in enumerate(measures):
    res = np.load('results/%s.npy' % m)

    for d_id, res_drift in enumerate(res):
        for r_id, res_rep in enumerate(res_drift):
            #shuffle
            p = np.random.permutation(res_rep.shape[0])
            res_rep = res_rep[p]
        
            # print(res_rep.shape) # chunks, measures + label
            X = res_rep[:,:-1]
            y = res_rep[:,-1]
            
            X[np.isnan(X)]=1
            X[np.isinf(X)]=1
            
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3242)
            for fold, (train, test) in enumerate(rskf.split(X, y)):
            
                for base_id, base_c in enumerate(base_clfs):
                    clf = clone(base_c)
                    
                    pred = clf.fit(X[train], y[train]).predict(X[test])
                    acc = balanced_accuracy_score(y[test], pred)
                    
                    clf_res[m_id, d_id, r_id, fold, base_id] = acc
                    pbar.update(1)
                
            print(m, np.mean(clf_res[m_id, d_id, r_id], axis=0))
            
    np.save('results/clf.npy', clf_res)
