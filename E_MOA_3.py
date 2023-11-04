"""
E3, E4 - select k-best + classification + f-test anova --- MOA streams
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
import os

np.random.seed(1233)

base_clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(random_state=11313),
    DecisionTreeClassifier(random_state=11313),
    MLPClassifier(random_state=11313)
]

streams = os.listdir('data/moa')
streams.remove('.DS_Store')
print(streams)

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]

n_splits=2
n_repeats=5


for f_id in range(len(streams)):
    
    clf_res = np.zeros((len(n_features), n_splits*n_repeats, len(base_clfs)))
    anova_res = np.zeros((max(n_features), 2))

    pbar = tqdm(total=len(n_features)*n_splits*n_repeats*len(base_clfs))

    res_temp = np.load('results/combined_moa_%i.npy' % f_id)
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
               
    np.save('results/clf_sel_moa_%i.npy' % f_id, clf_res)
    np.save('results/anova_sel_moa_%i.npy' % f_id, anova_res)
    
