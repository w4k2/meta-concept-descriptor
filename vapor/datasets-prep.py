import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.io import arff
import matplotlib.pyplot as plt

data = arff.loadarff('arffdata/Sudden_0.arff')
df = pd.DataFrame(data[0])

df.to_csv('arffdata/test.csv')

X = df.to_numpy()
y = X[:,-1]
X = X[:,:-1]
print(X.shape)

X[np.isnan(X)]=1

X_embedded = TSNE(n_components=2).fit_transform(X)

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='jet')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('foo.png')