import pandas as pd
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from googletrans import Translator


da = pd.read_csv('cora/cora_net.emb', index_col=[0])

a = np.array(np.array(da), dtype=float)
pred = KMeans(n_clusters=7).fit_predict(a)
da['cus_id'] = pred
da['cus_id'].value_counts()

group = [i.strip('\n') for i in open('cora/group.txt')]
group1 = pd.DataFrame(group)

data = da.join(group1)

a = np.array(data.iloc[:, 0:100])

xy = TSNE(2).fit_transform(a)
data['x'] = xy[:,0]
data['y'] = xy[:,1]

cname = ['black','blue','brown', 'red', 'pink', 'green', 'yellow']

type1 = list(set(data[0].values))

ax = plt.figure().add_subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

for i, n in zip(type1, cname):
    d = data[data[0] == i]
    ax.scatter(list(d['x']), list(d['y']), c=n, alpha=0.8, s=5, label=str(i))
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1), prop={'size':8})
# plt.gcf().savefig('thuc_without.png')
plt.show()