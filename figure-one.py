'''
Script generating sample figure for manuscript
'''

import matplotlib.pyplot as plt
import numpy as np
from strlearn.streams import StreamGenerator

stream = StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, n_chunks=250, chunk_size=1)
_ = stream._make_classification()

scp = 1-stream.concept_probabilities
tpoints = np.linspace(0,1,stream.n_chunks)

static_end = np.where(scp > .1)[0][0]
early_end = np.where(scp > .25)[0][0]
central_end = np.where(scp > .75)[0][0]
late_end = np.where(scp > .9)[0][0]
ends = [static_end, early_end, central_end, late_end]

textpoints = np.array([
    static_end/2,
    static_end + (early_end-static_end)/2,
    stream.n_chunks//2,
    late_end + (central_end-late_end)/2,
    stream.n_chunks - static_end/2
]).astype(int)
texts = [
    'static',
    'early\ndrift',
    'central\ndrift',
    'late\ndrift',
    'static'
]

print(static_end, early_end, central_end, late_end)

# Establish figure
fig, ax = plt.subplots(1,1,figsize=(8,8/1.618))

ax.set_yticks([0,.1,.25,.75,.9,1])
ax.set_xticks(np.linspace(0,1,11))
ax.plot(tpoints, scp, c='black')
ax.fill_between(tpoints, scp, tpoints*0, color='#DDD')
#ax.vlines(tpoints[ends], 0, 1, color='red')

for tpe, sc in zip(tpoints[ends], scp[ends]):
    print(tpe, sc)
    ax.plot([tpe,tpe], [0, sc], c='red', ls=":")
    ax.plot([tpe,tpe], [sc, 1], c='red')

for idx, (tp, tt, t) in enumerate(zip(tpoints[textpoints], textpoints, texts)):
    sp = scp[tt]
    ax.text(tp, .5, t, ha='center', va='center',
            bbox=dict(boxstyle="round",
                   ec=(1., 1, 1),
                   fc='#FCC',
                   ), fontsize=14)
    
ax.set_ylim(0,1)
ax.set_xlim(0,1)

ax.grid(ls=":")
ax.set_ylabel('emerging concept probability')
ax.set_xlabel('emerging concept flow')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/figure1.eps')