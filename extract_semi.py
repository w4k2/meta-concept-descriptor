"""
Przygotowanie strumieni i meta - semi
"""

import numpy as np
from vapor.SSG import SemiSyntheticStreamGenerator
from pymfe.mfe import MFE
from tqdm import tqdm

replications = 1
random_states = np.random.randint(100,10000,replications)

drift_types = {
    'nearest': {'interpolation':'nearest', 'n_drifts': 20},
    'cubic': {'interpolation':'cubic', 'n_drifts': 6},
}

static_data = ['static_data/australian.csv',
               'static_data/banknote.csv',
               'static_data/diabetes.csv',
               'static_data/german.csv',
               'static_data/vowel0.csv',
               'static_data/wisconsin.csv'
               ]

stream_static = {
                'n_chunks': 5000,
                'chunk_size': 200,
                'n_features': 10,        
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

pbar = tqdm(total=len(static_data) * len(drift_types) * len(random_states) * stream_static['n_chunks'])

for m_id, measure_key in enumerate(measures):
    
    out = np.zeros((len(static_data), len(drift_types), len(random_states), stream_static['n_chunks'], measures_len[m_id]+1))
    pbar.reset()
    print(measure_key)
    for dataset_id, dataset in enumerate(static_data):
        
        data = np.loadtxt(dataset, delimiter=',')
        X_static = data[:,:-1]
        y_static = data[:,-1]
        
        for dt_id, dt in enumerate(drift_types):
            for rep, rs in enumerate(random_states):

                config = {
                    'X':X_static,
                    'y':y_static,
                    **stream_static,
                    **drift_types[dt],
                    'random_state': rs
                }
                            
                stream = SemiSyntheticStreamGenerator(**config)
                stream._make_stream()
                
                e = stream._concept_proba()[::stream_static['chunk_size']]           
                drifts = stream._get_drifts()
                            
                concept=0
                decreasing = False
                
                for chunk in range(stream_static['n_chunks']): 
                    
                    #IDENTIFY CONCEPT
                    if dt=='nearest':
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
                    
                    out[dataset_id, dt_id, rep, chunk, :-1] = ft
                    out[dataset_id, dt_id, rep, chunk, -1] = concept
                    pbar.update(1)
                    
        np.save('res/semi_%s.npy' % measure_key, out)
