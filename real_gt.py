"""
Oznaczanie momentów dryfu w rzeczywistych
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import strlearn as sl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from utils import ELMI
real_streams = [
    'real_streams/covtypeNorm-1-2vsAll-pruned.arff',
    'real_streams/electricity.npy',
    'real_streams/poker-lsn-1-2vsAll-pruned.arff',
    'real_streams/INSECTS-abrupt_imbalanced_norm.arff',
    'real_streams/INSECTS-gradual_imbalanced_norm.arff',
    'real_streams/INSECTS-incremental_imbalanced_norm.arff'
    ]

stream_static = { 'chunk_size': 300 }


pbar = tqdm(total=len(real_streams))


for f_id, f in enumerate(real_streams):
    out = []

    fname=(f.split('/')[1]).split('.')[0]

    if 'npy' in f:
        stream = sl.streams.NPYParser(f, chunk_size=stream_static['chunk_size'], n_chunks=100000)
    else:
        stream = sl.streams.ARFFParser(f, chunk_size=stream_static['chunk_size'], n_chunks=100000)

    X_all = []
    y_all = []
    for chunk in range(100000):    
        # Pruning
        try:
            X, y = stream.get_chunk()
        except Exception as e:
            print(e)
            break
        
        if len(X_all)>0:
            if X.shape != X_all[-1].shape:
                continue
 
        if len(np.unique(y))<2:
            print('skip', chunk, fname)
            continue
                            
        X_all.append(X)
        y_all.append(y)    
    
    _chunks = len(y_all)
    print(_chunks)

    # X_all = np.array(X_all)
    # y_all = np.array(y_all)
    # X_all = X_all.reshape(-1, X_all.shape[-1])
    # y_all = y_all.flatten()
    # print(X_all.shape)
    # print(y_all.shape)
    
    # all = np.concatenate((X_all, y_all[:,None]), axis=1)
    # print(all.shape)

    # np.save('real_streams_pr/%s.npy' % fname, all)
    
    stream = sl.streams.NPYParser('real_streams_pr/%s.npy' % fname, chunk_size=stream_static['chunk_size'], n_chunks=_chunks)
    
    if f_id==0:
        drfs=[57,121,131,155,205,260,295,350]
    if f_id==1:
        drfs=[20,38,55,115,145]
    if f_id==2:
        drfs=[45,90,110,120,160,182,245,275,292,320,358,400,450,468,480,516,540,550,590,600,640,710,790,831,850,880,900,920,965,1000,1010]
    if f_id==3:
        drfs=[125]
    if f_id==4:
        drfs=[9,60,90,125,190]
    if f_id==5:
        drfs=[9,35,60,180,220]
        
    # clf = [GaussianNB(), MLPClassifier()]
    # clf = [GaussianNB(), MLPClassifier(), ELMI(probing_rate=1., update_rate=1.)]
    clf = [GaussianNB(), MLPClassifier(), DecisionTreeClassifier()]
    
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    
    for i in range(len(clf)):
        plt.plot(evaluator.scores[i,:,1], alpha=0.3, label=['gnb', 'mlp', 'elm9', 'elm1'][i])
    plt.vlines(drfs,0.5,1, color='r')
    # plt.xticks(np.linspace(0,250,30))
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('real_streams_gt/%s.png' % fname)
    
    np.save('real_streams_gt/%s.npy' % fname, drfs)
    np.save('real_streams_gt/clf_%s.npy' % fname, evaluator.scores)
    
    # exit()
