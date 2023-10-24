"""
E3, E4 - selekcja k-best + klasyfikacja + zapis f-test anova - real
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
from sklearn.feature_selection import SelectKBest, f_classif
np.random.seed(1233)

base_clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(random_state=11313),
    DecisionTreeClassifier(random_state=11313),
    MLPClassifier(random_state=11313)
]
real_streams_full = [
    'real_streams/covtypeNorm-1-2vsAll-pruned.arff',
    'real_streams/electricity.npy',
    'real_streams/poker-lsn-1-2vsAll-pruned.arff',
    'real_streams/INSECTS-abrupt_imbalanced_norm.arff',
    'real_streams/INSECTS-gradual_imbalanced_norm.arff',
    'real_streams/INSECTS-incremental_imbalanced_norm.arff'
    ]

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]

n_splits=2
n_repeats=5


for f_id in range(len(real_streams_full)):
    
    clf_res = np.zeros((len(n_features), n_splits*n_repeats, len(base_clfs)))
    anova_res = np.zeros((max(n_features), 2))

    pbar = tqdm(total=len(n_features)*n_splits*n_repeats*len(base_clfs))

    res_temp = np.load('res_clf_cls/combined_real_%i.npy' % f_id)
    print(res_temp.shape) # features, chunks
    
    res_temp = res_temp.swapaxes(0,1)

    #shuffle
    p = np.random.permutation(res_temp.shape[0])
    res_temp = res_temp[p]

    # print(res_rep.shape) # chunks, measures + label
    X = res_temp[:,:-1]
    y = res_temp[:,-1]
    
    X[np.isnan(X)]=1
    X[np.isinf(X)]=1
    
    # print(X.shape)
    # print(y)
    
    stat, val = f_classif(X,y)
    anova_res[:,0] = stat
    anova_res[:,1] = val

    for n_id, n_f in enumerate(n_features):
        #selekcja
        skb = SelectKBest(k=n_f)
        X_new = skb.fit_transform(X, y)

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3242)
        for fold, (train, test) in enumerate(rskf.split(X_new, y)):
        
            for base_id, base_c in enumerate(base_clfs):
                clf = clone(base_c)
                
                pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
                acc = balanced_accuracy_score(y[test], pred)
                
                clf_res[n_id, fold, base_id] = acc
                # print(acc)
                pbar.update(1)
               
    np.save('res_clf_cls/clf_sel_real_%i.npy' % f_id, clf_res)
    np.save('res_clf_cls/anova_sel_real_%i.npy' % f_id, anova_res)
    
