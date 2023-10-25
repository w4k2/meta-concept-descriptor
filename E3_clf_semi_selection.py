"""
E3, E4 - select k-best + classification + f-test anova --- semi-synthetic streams
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

def sqspace(start, end, num):
    space = (((np.power(np.linspace(0,1,num),2))*(end-start))+start).astype(int)[1:]
    return space

n_features = sqspace(1,118,31)[1:]


origial_datasets=6

n_splits=2
n_repeats=5

n_drift_types = 2

clf_res = np.zeros((origial_datasets, n_drift_types, len(n_features), n_splits*n_repeats, len(base_clfs)))
anova_res = np.zeros((origial_datasets, n_drift_types, max(n_features), 2))

pbar = tqdm(total=origial_datasets*n_drift_types*len(n_features)*n_splits*n_repeats*len(base_clfs))

res = np.load('results/combined_semi.npy')
print(res.shape) # features, datasets, drift types, chunks
# exit()

for origin_id in range(origial_datasets):
    for d_id in range(n_drift_types):
        
        res_temp = res[:,origin_id,d_id]
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
        anova_res[origin_id, d_id,:,0] = stat
        anova_res[origin_id, d_id,:,1] = val

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
                    
                    clf_res[origin_id, d_id, n_id, fold, base_id] = acc
                    # print(acc)
                    pbar.update(1)
            
        print(np.mean(clf_res[origin_id, d_id], axis=1))
        
np.save('results/clf_sel_semi.npy', clf_res)
np.save('results/anova_sel_semi.npy', anova_res)
        
