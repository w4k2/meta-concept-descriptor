"""
Visualization for E4
"""
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

base_clfs = ['GNB','KNN','SVM','DT','MLP']

static_data = ['australian',
    'banknote',
    'diabetes',
    'german',
    'vowel0',
    'wisconsin'
    ]

real_streams = [
    'covtypeNorm-1-2vsAll',
    'electricity',
    'poker-lsn-1-2vsAll',
    'INSECTS-abrupt',
    'INSECTS-gradual',
    'INSECTS-incremental'
    ]

"""
Synthetic
"""
clf = np.load('results/clf_sel.npy')
reduced = np.load('results/clf_reduced.npy')
print(reduced.shape) # 3, 5, 10, 5

reduced_mean = np.mean(reduced, axis=(1,2))

fig, ax = plt.subplots(3, 1, figsize=(8,8), sharex=True, sharey=True)
    
for drf_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']):    
    img = np.zeros((2,5))
    full = np.mean(clf[drf_id, :, -1,:,:], axis=(0,1))
    reduced = reduced_mean[drf_id]
    img[0] = full
    img[1] = reduced-full
    
    ax[drf_id].imshow(img, vmin=0.05, vmax=1, cmap='Blues')
    ax[drf_id].set_title('%s drift' % drift_type)
    
    ax[drf_id].set_xticks(range(len(base_clfs)), base_clfs)
    ax[drf_id].set_yticks(range(2), ['full', 'reduced'])
    
    for _a, __a in enumerate(['full', 'reduced']):
        for _b, __b in enumerate(base_clfs):
            if _a==0:
                ax[drf_id].text(_b, _a, "%.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
            else:
                ax[drf_id].text(_b, _a, "%+.3f" % (img[_a, _b]) , va='center', ha='center', c='black' if img[_a, _b]<0.5 else 'white', fontsize=11)
    
plt.tight_layout()
plt.savefig('figures/reduced_syn.png')
plt.savefig('foo.png')

"""
Common
"""
fig, ax = plt.subplots(4,1,figsize=(13, 13))

# Synthetic
clf = np.load('results/clf_sel.npy')
reduced = np.load('results/clf_reduced.npy')

margin = .3
w = .05

c = plt.cm.turbo(np.linspace(0,1,6))
c2 = np.copy(c)
c2[:,:3] = c2[:,:3]/2+.5
print(c)

for drf_id, drift_type in enumerate(['Sudden', 'Gradual', 'Incremental']): 
    full = np.mean(clf[drf_id, :, -1,:,:], axis=(0,1))   
    reduced = reduced_mean[drf_id]    
    
    x = np.linspace(-margin, margin, len(full)) + drf_id
    
    ax[0].bar(x-w/2, full, width=w, color=c)
    ax[0].bar(x+w/2, reduced, width=w, color=c2)

ax[0].set_xticks([0,1,2], ['Sudden', 'Gradual', 'Incremental'])


legend_labels=[ '%s-%s' % (bc, d) for bc in base_clfs for d in ['F', 'R']]
legend_colors=[]
for i in range(len(c)):
    legend_colors.append(c[i])
    legend_colors.append(c2[i])
custom_lines = [Line2D([0], [0], color=legend_colors[i], lw=4) for i in range(len(legend_colors))]
ax[0].legend(custom_lines, legend_labels, ncol=5, frameon=False, loc=5, fontsize=11)

# Semi
reduced = np.load('results/semi_clf_reduced.npy')
clf = np.load('results/clf_sel_semi.npy')
reduced_mean = np.mean(reduced, axis=2)

for drf_id in range(2):
    for dataset_id, dataset in enumerate(static_data):
        full = np.mean(clf[dataset_id, drf_id, -1], axis=0)
        reduced = reduced_mean[dataset_id,drf_id]

        x = np.linspace(-margin, margin, len(full)) + dataset_id

        ax[1+drf_id].bar(x-w/2, full, width=w, color=c)
        ax[1+drf_id].bar(x+w/2, reduced, width=w, color=c2)
        
    ax[1+drf_id].set_xticks([0,1,2,3,4,5], static_data)

# Real    
res = np.load('results/real_clf_reduced.npy')
for f_id, f in enumerate(real_streams):
    clf = np.load('results/clf_sel_real_%i.npy' % f_id)    

    reduced = np.mean(res[f_id], axis=0)
    full = np.mean(clf[-1], axis=0)

    x = np.linspace(-margin, margin, len(full)) + f_id

    ax[3].bar(x-w/2, full, width=w, color=c)
    ax[3].bar(x+w/2, reduced, width=w, color=c2)
    
ax[3].set_xticks([0,1,2,3,4,5], real_streams)
    
# Styling
ax[0].set_title('Synthetic data streams')
ax[1].set_title('Semi-synthetic data streams with nearest interpolation')
ax[2].set_title('Semi-synthetic data streams with cubic interpolation')
ax[3].set_title('Real data streams')
ax[0].set_xlim(ax[-1].get_xlim())

for aids, aa in enumerate(ax.ravel()):
    aa.set_ylim(0,1)
    aa.grid(axis='y', ls=':')
    
    aa.set_ylabel('balanced accuracy score')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/reduced.png')
plt.savefig('figures/reduced.eps')
