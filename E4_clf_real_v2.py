"""
E4 - classification with reduced number of measures -- real-world streams
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
import utils

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

real_streams_full = [
    'data/real_streams/covtypeNorm-1-2vsAll-pruned.arff',
    'data/real_streams/electricity.npy',
    'data/real_streams/poker-lsn-1-2vsAll-pruned.arff',
    'data/real_streams/INSECTS-abrupt_imbalanced_norm.arff',
    'data/real_streams/INSECTS-gradual_imbalanced_norm.arff',
    'data/real_streams/INSECTS-incremental_imbalanced_norm.arff'
    ]

base_clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(random_state=11313),
    DecisionTreeClassifier(random_state=11313),
    MLPClassifier(random_state=11313)
]

origial_datasets=len(real_streams_full)

n_splits=2
n_repeats=5

clf_res = np.zeros((origial_datasets, n_splits*n_repeats, len(base_clfs)))
pbar = tqdm(total=origial_datasets*n_splits*n_repeats*len(base_clfs))

indexes = utils.selected2_indexes
indexes.append(-1) #keep label
labels = utils.selected2_measure_names

for f_id in range(len(real_streams_full)):
    res = np.load('results/combined_real_%i.npy' % f_id)

    res = res[indexes]
    res = res.swapaxes(0,1)
    
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
            
            clf_res[f_id, fold, base_id] = acc
            pbar.update(1)
        
    print(np.mean(clf_res[f_id], axis=0))
    
np.save('results/real_clf_reduced.npy', clf_res)
