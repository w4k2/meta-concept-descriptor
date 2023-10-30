import numpy as np
import matplotlib.pyplot as plt

from utils import find_real_drift
from pymfe.mfe import MFE
from strlearn.streams import ARFFParser


chunk_size = 200
n_chunks = 5000

n_drifts=4
# stream = np.load('data/moa/stream.npy')

drfs = find_real_drift(n_chunks,n_drifts)
print(drfs)

#### EXPERIMENT 1 -- PREPARE ####

concept=0

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


for m_id, measure_key in enumerate(measures):
    print(measure_key)
    concept=0
    out = []
    
    stream = ARFFParser(path='data/moa/sd_s_sea_r1_s_sea_r2.arff', chunk_size=chunk_size)
    
    for chunk in range(n_chunks):
        if chunk%100 == 0:
            print(chunk)

        if chunk in drfs:
            concept+=1 #chunk 125 (w drift) to juz nowa koncepcja
                        
        # CALCULATE
        # data = stream[chunk*chunk_size : (chunk+1)*chunk_size]
        try:
            X, y = stream.get_chunk()
        except:
            break
        #data[:,:-1], data[:,-1]
        # print(X.shape, y.shape)
        # exit()
        
        mfe = MFE(groups=[measure_key])
        mfe.fit(X,y)
        ft_labels, ft = mfe.extract()
        ft.append(concept%2)

        out.append(ft)
            
    np.save('results/moa_%s.npy' % (measure_key), np.array(out))

