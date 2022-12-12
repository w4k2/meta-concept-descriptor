"""
E4 - klasyfikacja syntetycznych, ale tylko dla 17 wybranych metryk
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
from sklearn.metrics import accuracy_score
import utils

np.random.seed(1233)

base_clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(random_state=11313),
    DecisionTreeClassifier(random_state=11313),
    MLPClassifier(random_state=11313)
]

n_splits=2
n_repeats=5

n_drift_types=3
stream_reps=5

clf_res = np.zeros((n_drift_types, stream_reps, n_splits*n_repeats, len(base_clfs)))
pbar = tqdm(total=n_drift_types*stream_reps*n_splits*n_repeats*len(base_clfs))

indexes = utils.selected2_indexes
indexes.append(-1) #keep label
labels = utils.selected2_measure_names

res = np.load('res_clf_cls/combined_syn.npy')
print(res.shape) # features+label, drifts, reps, chunks

res = res[indexes]
print(res.shape)

for d_id in range(n_drift_types):
    for r_id in range(stream_reps):
        #shuffle
        res_temp = res[:,d_id,r_id]
        res_temp = res_temp.swapaxes(0,1)

        p = np.random.permutation(res_temp.shape[0])
        res_temp = res_temp[p]
    
        # print(res_rep.shape) # chunks, measures + label
        X = res_temp[:,:-1]
        y = res_temp[:,-1]

        X[np.isnan(X)]=1
        X[np.isinf(X)]=1
        
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3242)
        for fold, (train, test) in enumerate(rskf.split(X, y)):
        
            for base_id, base_c in enumerate(base_clfs):
                clf = clone(base_c)
                
                pred = clf.fit(X[train], y[train]).predict(X[test])
                acc = accuracy_score(y[test], pred)
                
                clf_res[d_id, r_id, fold, base_id] = acc
                pbar.update(1)
            
        print(np.mean(clf_res[d_id, r_id], axis=0))
        
np.save('res_clf_cls/clf_reduced.npy', clf_res)
