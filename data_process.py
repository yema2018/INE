import os
import json
import re
import pandas as pd
import re

"""
cla_ml = [(i.split('\t')[0], i.split('\t')[1]) for i in open('cora/classifications') if 'Machine_Learning' in i]

claml1 = [j[0] for j in cla_ml]

ml_node = [i for i in open('cora/papers') if i.split('\t')[1] in claml1]

ml_dict = {}
for m in ml_node:
    ml_dict = {}
    url = m.split('\t')[1]
    ml_dict['id'] = m.split('\t')[0]
    ml_dict['type'] = cla_ml[claml1.index(url)][-1]
    try:
        title = re.findall('<title> (.*?) </title>', m)[0]
    except:
        continue
    try:
        booktitle = re.findall('<booktitle> (.*?) </booktitle>', m)[0]
    except:
        booktitle = 'None'
    try:
        category = re.findall('<type> (.*?) </type>', m)[0]
    except:
        category = 'None'
    ml_dict['title'] = title
    ml_dict['category'] = category
    ml_dict['booktitle'] = booktitle
    with open('cora.json', 'a', encoding='utf-8') as fw:
        json.dump(ml_dict, fw, ensure_ascii=False)
        fw.write('\n')


a = [json.loads(i) for i in open('cora.json')]
keys = ['category', 'type', 'title', 'booktitle', 'id']

da = [(i[keys[0]], i[keys[1]], i[keys[2]], i[keys[3]], i[keys[4]]) for i in a]

import pandas as pd

daa = pd.DataFrame(da, columns=keys)
daa = daa.drop_duplicate



cora = [(i.strip('\n').split('\t')[0], i.strip('\n').split('\t')[1]) for i in open('cora/graph.txt')]

a = pd.DataFrame(cora)
a.to_csv('cora/cora.edge', sep= ' ', header=False, index=False)

abss = [i.strip('\n') for i in open('cora/data.txt')]
abs = pd.DataFrame(abss, index=range(len(abss)))

abs.to_csv('cora/cora.map', header=False)
"""


e = [i.strip(' \n').split() for i in open('M10/adjedges.txt')]

with open('M10/M10.edge', 'w') as fw:
    for a in e:
        if len(a) == 1:
            continue
        for i in range(1, len(a)):
            fw.write(a[0]+' '+a[i]+'\n')


e = [set(i.strip(' \n').split()) for i in open('M10/M10.edge')]
drop_id = []
for i in range(len(e)-1):
    try:
        a = e.index(e[i], (i+1))
    except: continue
    drop_id.append(a)


e = pd.read_csv('M10/M10.edge', sep=' ', names=['1','2'])
e = e.drop(drop_id)
e.to_csv('M10/M10.edge',sep=' ',header=False,index=False)


drop_id=[]
e = pd.read_csv('M10/M10.edge', sep=' ', names=['1','2'])
a = [i.strip('\n').split(' ') for i in open('M10/docs.txt')]
index = [i[0] for i in a]
index = [str(i) for i in mapf.index]
for i in range(len(e)):
    v = e.iloc[i].astype(str).values
    if v[0] not in index or v[1] not in index:
        drop_id.append(i)
e = e.drop(drop_id)
e.to_csv('M10/M10.edge',sep=' ',header=False,index=False)



a = [i.strip('\n').split(' ') for i in open('M10/docs.txt')]
index = [i[0] for i in a]
text = [re.sub('[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_'
               '`{|}~]+', "", ' '.join(i[1:])).strip() for i in a]
a = pd.DataFrame({'0':index, '1':text})
a.to_csv('M10/M10.map',index=False,header=False)


node = list(set(list(e['1'])+list(e['2'])))
node = [str(i) for i in node]
mapf.index = [str(i) for i in mapf.index]
map = mapf.loc[node]
map.to_csv('M10/M10.map',header=False)


g = pd.read_csv('M10/group.txt', index_col=[0], sep=' ', names='id')
g = g.loc[node]
g.to_csv('M10/group.txt',sep=' ',header=False)