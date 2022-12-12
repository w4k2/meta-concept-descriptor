"""
E3, E4 - selekcja k-best + klasyfikacja + zapis f-test anova - syntetyczne
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
from sklearn.feature_selection import SelectKBest, f_classif
np.random.seed(1233)

base_clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(random_state=11313),
    DecisionTreeClassifier(random_state=11313),
    MLPClassifier(random_state=11313)
]

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]

n_splits=2
n_repeats=5

n_drift_types = 3
stream_reps = 5

clf_res = np.full((n_drift_types, stream_reps, len(n_features), n_splits*n_repeats, len(base_clfs)), np.nan).astype(float)
anova_res = np.full((n_drift_types, stream_reps, max(n_features), 2), np.nan).astype(float)

pbar = tqdm(total=n_drift_types*len(n_features)*stream_reps*n_splits*n_repeats*len(base_clfs))

res = np.load('res_clf_cls/combined_syn.npy')
print(res.shape) # drfs, reps, chunks, measures + label

for d_id in range(n_drift_types):    
    for r_id in range(stream_reps):
        res_temp = res[:,d_id,r_id]
        res_temp = res_temp.swapaxes(0,1)

        #shuffle
        p = np.random.permutation(res_temp.shape[0])
        res_temp = res_temp[p]
    
        # print(res_rep.shape) # chunks, measures + label
        X = res_temp[:,:-1]
        y = res_temp[:,-1]
        
        X[np.isnan(X)]=1
        X[np.isinf(X)]=1
        
        print(X.shape)
        print(y)
        
        stat, val = f_classif(X,y)
        anova_res[d_id,r_id,:,0] = stat
        anova_res[d_id,r_id,:,1] = val

        for n_id, n_f in enumerate(n_features):
            #selekcja
            skb = SelectKBest(k=n_f)
            X_new = skb.fit_transform(X, y)

            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3242)
            for fold, (train, test) in enumerate(rskf.split(X_new, y)):
            
                for base_id, base_c in enumerate(base_clfs):
                    clf = clone(base_c)
                    
                    pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
                    acc = accuracy_score(y[test], pred)
                    
                    clf_res[d_id, r_id, n_id, fold, base_id] = acc
                    # print(acc)
                    pbar.update(1)
            
        print(np.mean(clf_res[d_id, r_id], axis=1))
        
np.save('res_clf_cls/clf_sel.npy', clf_res)
np.save('res_clf_cls/anova_sel.npy', anova_res)
        
