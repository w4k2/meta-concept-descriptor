import numpy as np
import matplotlib.pyplot as plt
from skmultiflow.data import ConceptDriftStream, AGRAWALGenerator, MIXEDGenerator
from utils import find_real_drift
from sklearn.neural_network import MLPClassifier
from scipy.signal import medfilt

chunk_size = 200
n_drifts=20
n_chunks = 500
# stream = ConceptDriftStream(random_state=677, position = 200*200, width = 100*200)
# stream._random_state = np.random
stream = AGRAWALGenerator(random_state=111)
# stream = MIXEDGenerator(random_state=222)

m = []

for i in range(int(n_chunks)):
    X, y = stream.next_sample(chunk_size)
    if i==100:
        stream.generate_drift()
    
    # print(X.shape, np.array(y).shape) # chunks, instances, features
    m.append(np.mean(X, axis=0))
    
m = np.array(m)
print(m.shape)
plt.plot(medfilt(m,15))
plt.savefig('foo.png')