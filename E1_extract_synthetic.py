"""
Experiment 1 -- collect streams and metafetaures -- synthetic
"""

import numpy as np
import strlearn as sl
from pymfe.mfe import MFE
from utils import find_real_drift
from tqdm import tqdm

replications = 5
random_states = np.random.randint(100,10000,replications)

drift_types = {
    'sudden': {'concept_sigmoid_spacing':9999, 'n_drifts': 20,},
    'gradual': {'concept_sigmoid_spacing':5, 'n_drifts': 6},
    'incremental': {'concept_sigmoid_spacing':5, 'incremental':True, 'n_drifts': 6}
}

stream_static = {
                'n_chunks': 5000,
                'chunk_size': 200,
                'n_features': 10,
                'n_informative': 10,
                'n_redundant': 0,
                'class_sep':1,
            }

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

measures_len = [
    8,
    35,
    8, 
    12,
    13,
    4,
    14,
    24,
    48
]

pbar = tqdm(total=len(drift_types) * len(random_states) * stream_static['n_chunks'])

for m_id, measure_key in enumerate(measures):
    
    out = np.zeros((len(drift_types), len(random_states), stream_static['n_chunks'], measures_len[m_id]+1))
    pbar.reset()
    print(measure_key)
    for dt_id, dt in enumerate(drift_types):
        for rep, rs in enumerate(random_states):

            config = {
                **stream_static,
                **drift_types[dt],
                'random_state': rs
            }
            
            drifts = find_real_drift(config['n_chunks'], config['n_drifts'])
            
            stream = sl.streams.StreamGenerator(**config)
            e = stream._sigmoid(
                stream.concept_sigmoid_spacing, stream.n_drifts
            )[1][::stream_static['chunk_size']]
                        
            concept=0
            decreasing = True

            for chunk in range(stream_static['n_chunks']): 
                
                #IDENTIFY CONCEPT
                if dt=='sudden':
                    if chunk in drifts:
                        concept+=1 #chunk 125 (w drift) to juz nowa koncepcja
                else:
                    if decreasing:
                        if concept%4==0:
                            if e[chunk]<0.9:
                                concept+=1
                        if concept%4==1:
                            if e[chunk]<0.75:
                                concept+=1
                        if concept%4==2:
                            if e[chunk]<0.25:
                                concept+=1
                        if concept%4==3:
                            if e[chunk]<0.1:
                                concept+=1
                                decreasing = False
                    else:
                        if concept%4==0:
                            #szukamy punktu 0.1
                            if e[chunk]>0.1:
                                concept+=1
                        if concept%4==1:
                            #szukamy punktu 0.25
                            if e[chunk]>0.25:
                                concept+=1
                        if concept%4==2:
                            #szukamy punktu 0.75
                            if e[chunk]>0.75:
                                concept+=1
                        if concept%4==3:
                            #szukamy punktu 0.9
                            if e[chunk]>0.9:
                                concept+=1
                                decreasing = True


                # CALCULATE
                X, y = stream.get_chunk()
                
                mfe = MFE(groups=[measure_key])
                mfe.fit(X,y)
                ft_labels, ft = mfe.extract()
                # print(ft_labels, len(ft))
                
                out[dt_id, rep, chunk, :-1] = ft
                out[dt_id, rep, chunk, -1] = concept
                pbar.update(1)
                
    np.save('res/%s.npy' % measure_key, out)
