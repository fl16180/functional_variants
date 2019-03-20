''' generates figures used in project report.
Requires seaborn and matplotlib which are not required in environment. '''


from os.path import join
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config as cfg
from model.data_loaders import load_train_set
from utils.data_utils import downsample_negatives
from utils.metrics import avgPR


res = pd.read_csv(join(cfg.OUTPUT_DIR, 'scores.csv'))

res = downsample_negatives(res, p=0.2)

# tsne
sub_res = res.iloc[:,15:1000]
scaler = StandardScaler()
sub_res = scaler.fit_transform(sub_res)

tsne = TSNE(n_components=2, verbose=1, perplexity=20.0, n_iter=500, learning_rate=50)
tsne_res = tsne.fit_transform(sub_res)

df_tsne = res.copy()
df_tsne['tsne-1'] = tsne_res[:,0]
df_tsne['tsne-2'] = tsne_res[:,1]
df_tsne['Predicted'] = (df_tsne['Score'] > 0.03).astype(int)

tsne1 = sns.lmplot(x='tsne-1', y='tsne-2', data=df_tsne, hue='Label', fit_reg=False, palette='Set2')
plt.title('t-SNE components by true label')
fig = tsne1.fig
fig.savefig('./f1.png', dpi=150, bbox_inches='tight')

tsne2 = sns.lmplot(x='tsne-1', y='tsne-2', data=df_tsne, hue='Predicted', fit_reg=False, palette='Set2')
plt.title('t-SNE components by predicted label, threshold=0.03')
fig = tsne2.fig
fig.savefig('./f2.png', dpi=150, bbox_inches='tight')




# error analysis
res = pd.read_csv(join(cfg.OUTPUT_DIR, 'scores.csv'))

train = load_train_set('E116')
train_counts = train.groupby('chr').sum().Label

errs = np.zeros(22)
for c in range(1,23):
    tmp = res[res['chr'] == c]
    met = avgPR(tmp['Label'], tmp['Score'])
    errs[c-1] = met

chrs = np.arange(1,23)
col1 = '#66c2a5'
col2 = '#fc8d62'

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].bar(chrs, errs, color=col1)
axes[1].bar(chrs, train_counts, color=col2)
axes[1].invert_yaxis()

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(True)
axes[0].set_ylabel('AUPR')
axes[0].set_title('AUPR and # positive train examples, by chromosome')
axes[0].set_ylim([0,1])
axes[0].set_yticks([0,1])
axes[0].tick_params(axis='x', length=0, width=0, labelsize=11)
axes[0].set_xlim([0.5,22.5])
axes[0].set_xticks([5,10,15,20])


axes[1].spines['bottom'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_ylabel('# positive class')
axes[1].set_ylim([60,0])
axes[1].set_yticks([60,0])
axes[1].tick_params(axis='x', length=0, width=0, labelsize=11)
fig.set_size_inches(8,5)
fig.savefig('./err.png', dpi=150, bbox_inches='tight')
