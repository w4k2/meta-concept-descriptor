import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from detectors.ADWIN import ADWIN
from detectors.meta import Meta
from tqdm import tqdm
import strlearn as sl
import matplotlib.pyplot as plt
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

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    X_all = X_all.reshape(-1, X_all.shape[-1])
    y_all = y_all.flatten()
    print(X_all.shape)
    print(y_all.shape)
    
    all = np.concatenate((X_all, y_all[:,None]), axis=1)
    print(all.shape)

    np.save('real_streams_pr/%s.npy' % fname, all)
    
    stream = sl.streams.NPYParser('real_streams_pr/%s.npy' % fname, chunk_size=stream_static['chunk_size'], n_chunks=_chunks)
    
    # clf =MLPClassifier()
    # clf = [GaussianNB(), MLPClassifier()]
    # # clf = Meta(base_clf=GaussianNB(), detector=ADWIN())
    
    # evaluator = sl.evaluators.TestThenTrain()
    # evaluator.process(stream, clf)
    # # print(evaluator.scores.shape)
    # exit()
    
    # fig, ax = plt.subplots(1,1,figsize=(20,10))
    
    # plt.plot(evaluator.scores[0,:,1])
    # plt.plot(evaluator.scores[1,:,1])
    # plt.xticks(np.linspace(0,250,30))
    # plt.grid()
    # plt.savefig('foo.png')
    
    if f_id==0:
        drfs=[55,120,130,155,260,295,300,320,330,350]
    if f_id==1:
        drfs=[20,38,60,70,115,145]
    if f_id==2:
        drfs=[45,90,110,120,160,182,245,275,292,320,358,400,450,468,480,516,540,550,590,600,640,710,790,831,850,880,900,920,965,1000,1010]
    if f_id==3:
        drfs=[]
    if f_id==4:
        drfs=[9,60,70,90,190]
    if f_id==5:
        drfs=[9,35,180,220]
        
    # drfs = np.argwhere(np.array(clf.detector.drift)==2).flatten()+1
    # print(drfs)
    np.save('real_streams_gt/%s.npy' % fname, drfs)
    
    # exit()
