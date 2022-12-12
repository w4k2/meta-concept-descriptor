"""
Przygotowanie strumieni i meta - rzeczywiste
"""

import numpy as np
from pymfe.mfe import MFE
from tqdm import tqdm
import strlearn as sl

real_streams = [
    'real_streams/covtypeNorm-1-2vsAll-pruned.arff',
    'real_streams/electricity.npy',
    'real_streams/poker-lsn-1-2vsAll-pruned.arff',
    'real_streams/INSECTS-abrupt_imbalanced_norm.arff',
    'real_streams/INSECTS-gradual_imbalanced_norm.arff',
    'real_streams/INSECTS-incremental_imbalanced_norm.arff'
    ]

stream_static = { 'chunk_size': 300 }

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

pbar = tqdm(total=len(real_streams))

for m_id, measure_key in enumerate(measures):
    print(measure_key)
    
    for f_id, f in enumerate(real_streams):
        fname=(f.split('/')[1]).split('.')[0]

        drfs = np.load('real_streams_gt/%s.npy' % fname)
        
        concept=0
        out = []
        
        stream = sl.streams.NPYParser('real_streams_pr/%s.npy' % fname, chunk_size=stream_static['chunk_size'], n_chunks=100000)
        
        for chunk in range(100000):
            # GET CONCEPT
            if chunk in drfs:
                concept+=1
                # print(concept)
                           
            # CALCULATE
            try:
                X, y = stream.get_chunk()
            except Exception as e:
                print(e)
                break
                    
            if len(np.unique(y))<2:
                # print('skip', chunk)
                continue
                # exit()
                                
            mfe = MFE(groups=[measure_key])
            mfe.fit(X,y)
            ft_labels, ft = mfe.extract()
            ft.append(concept)

            out.append(ft)
        # print(np.array(out).shape)
        # exit()
                
        np.save('res/real_%i_%s.npy' % (f_id, measure_key), np.array(out))
