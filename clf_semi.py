"""
E2 - klasyfikacja semi
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

origial_datasets=6

n_splits=2
n_repeats=5

n_drift_types = 2

clf_res = np.zeros((len(measures), origial_datasets, n_drift_types, n_splits*n_repeats, len(base_clfs)))

pbar = tqdm(total=len(measures)*origial_datasets*n_drift_types*n_splits*n_repeats*len(base_clfs))

for m_id, m in enumerate(measures):
    res = np.load('res/semi_%s.npy' % m)
    res = res.reshape(6,2,5000,-1)
    # print(res.shape) # drfs, reps, chunks, measures + label

    for origin_id, res_origin in enumerate(res):
        for d_id, res_drift in enumerate(res_origin):
            # print(res_drift.shape) # reps, chunks, measures + label

            p = np.random.permutation(res_drift.shape[0])
            res_drift = res_drift[p]
        
            # print(res_rep.shape) # chunks, measures + label
            X = res_drift[:,:-1]
            y = res_drift[:,-1]
            
            X[np.isnan(X)]=1
            X[np.isinf(X)]=1
            
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3242)
            for fold, (train, test) in enumerate(rskf.split(X, y)):
            
                for base_id, base_c in enumerate(base_clfs):
                    clf = clone(base_c)
                    
                    pred = clf.fit(X[train], y[train]).predict(X[test])
                    acc = balanced_accuracy_score(y[test], pred)
                    
                    clf_res[m_id, origin_id, d_id, fold, base_id] = acc
                    pbar.update(1)
                
            print(m, np.mean(clf_res[m_id, origin_id, d_id], axis=0))
        
np.save('res_clf_cls/semi_clf.npy', clf_res)
