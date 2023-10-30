import numpy as np
from utils import find_real_drift
from pymfe.mfe import MFE
from strlearn.streams import ARFFParser
import os


chunk_size = 200
n_chunks = int(100000/chunk_size)

n_drifts = 1
streams = os.listdir('data/moa')
streams.remove('.DS_Store')
print(streams)

drfs = find_real_drift(n_chunks,n_drifts)
print(drfs)

#### EXPERIMENT 1 ####

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

for s_id, s_name in enumerate(streams):
    for m_id, measure_key in enumerate(measures):
        print(measure_key, s_name)
        concept=0
        out = []
        
        stream = ARFFParser(path='data/moa/%s' % s_name, chunk_size=chunk_size)
        
        for chunk in range(n_chunks):
            if chunk%100 == 0:
                print(chunk)

            if chunk in drfs:
                concept+=1 #chunk 125 (w drift) to juz nowa koncepcja
                            
            # CALCULATE
            X, y = stream.get_chunk()
            
            mfe = MFE(groups=[measure_key])
            mfe.fit(X,y)
            ft_labels, ft = mfe.extract()
            ft.append(concept)

            out.append(ft)
                
        np.save('results/moa_%s_%i.npy' % (measure_key, s_id), np.array(out))

