import pandas as pd
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from googletrans import Translator
#
# fr = open('dblp/deepwalk.emb').readlines()[1:]
# emb = [i.strip('\n').split() for i in fr]
# emb = pd.DataFrame(emb)
# emb = emb.set_index([0]).astype(float)
# emb.columns = [str(i) for i in range(100)]
# emb.index = [int(i) for i in emb.index]
# da = emb

da = pd.read_csv('cora/b10/a0.9_3_0.8658431165906076', index_col=[0])

# a = np.array(np.array(da), dtype=float)
# pred = KMeans(n_clusters=7).fit_predict(a)
# da['cus_id'] = pred
# da['cus_id'].value_counts()

group1 = pd.read_csv('cora/group.txt',sep=' ',index_col=[0],names=[0])

data = da.join(group1)
# data = data.sample(frac=0.1)
# data = data.loc[shuffled[-int(len(data)*0.3):]]
a = np.array(data.iloc[:, 0:100])

xy = TSNE(2).fit_transform(a)
data['x'] = xy[:, 0]
data['y'] = xy[:, 1]

cname = ['black', 'blue', 'brown', 'red', 'pink', 'green', 'yellow']

l1 = list(data[0].values)
type1 = list(set(l1))
type1.sort(key=l1.index)

ax = plt.figure().add_subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height])

for i, n in zip(type1, cname):
    d = data[data[0] == i]
    ax.scatter(list(d['x']), list(d['y']), c=n, alpha=0.8, s=5, label=str(i))
    # ax.legend(loc='upper right', bbox_to_anchor=(1.3,1), prop={'size':8})
# plt.gcf().savefig('thuc_without.png')
plt.axis('off')
plt.show()

