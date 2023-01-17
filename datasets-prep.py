import numpy as np
import utils
import arff

# data = np.load('res_clf_cls/combined_syn.npy')
# print(data.shape)

# cols = utils.measure_labels_selected_flat
# cols = list(cols)
# cols.append('label')
# print(len(cols))

# print(data[:,1,1].shape)

# for dt_id, dt in enumerate(['Sudden', 'Gradual', 'Incremental']):
#     for rep in range(5):
#         arff.dump('arffdata/%s_%i.arff' % (dt, rep)
#             , data[:,dt_id,rep].T
#             , relation='Synthetic_Data_Stream_Metafeatures_Drift_Type:%s_Replication:%i' % (dt, rep)
#             , names=cols)

from scipy.io import arff
import pandas as pd

data = arff.loadarff('arffdata/Sudden_0.arff')
df = pd.DataFrame(data[0])

df.to_csv('arffdata/test.csv')

print(df)

from sklearn.manifold import TSNE

X = df.to_numpy()
y = X[:,-1]
X = X[:,:-1]
print(X.shape)

X[np.isnan(X)]=1

X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='jet')
ax.set_xticks([])
ax.set_yticks([])
# ax.grid(ls=":")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('foo.png')