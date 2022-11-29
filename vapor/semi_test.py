import numpy as np
from vapor.SSG import SemiSyntheticStreamGenerator
import matplotlib.pyplot as plt

drift_type = {'interpolation':'cubic', 'n_drifts': 6}

static_data = 'static_data/banknote.csv'

stream_static = {
                'n_chunks': 5000,
                'chunk_size': 200,
                'n_features': 2,        
            }

data = np.loadtxt(static_data, delimiter=',')
X_static = data[:,:-1]
y_static = data[:,-1]

config = {
    'X':X_static,
    'y':y_static,
    **stream_static,
    **drift_type,
    'random_state': 32
}
            
stream = SemiSyntheticStreamGenerator(**config)
stream._make_stream()

e = stream._concept_proba()[::stream_static['chunk_size']]           
drifts = stream._get_drifts()
            
concept=0
decreasing = False

perm = np.random.permutation(30)

map1 = plt.cm.jet(np.linspace(0,1,30))[perm]
map2 = plt.cm.autumn(np.linspace(0,1,30))[perm]

for chunk in range(stream_static['n_chunks']): 
    
    #IDENTIFY CONCEPT
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
    y = y.astype(int)
    colors = [map1[concept], map2[concept]]
    colors_y = np.array(colors)[y]
    
    plt.title('ch:%i, con:%i' % (chunk, concept))
    plt.scatter(X[:,0], X[:,1], c=colors_y)
    plt.ylim(-50,50)
    plt.xlim(-50,50)

    plt.tight_layout()
    plt.savefig('frames/%05d.png' % chunk)
    plt.savefig('foo.png')
    plt.clf()
    # exit()
