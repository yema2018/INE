import os
import json
import re
import pandas as pd

# cla_ml = [(i.split('\t')[0], i.split('\t')[1]) for i in open('cora/classifications') if 'Machine_Learning' in i]
#
# claml1 = [j[0] for j in cla_ml]
#
# ml_node = [i for i in open('cora/papers') if i.split('\t')[1] in claml1]
#
# ml_dict = {}
# for m in ml_node:
#     ml_dict = {}
#     url = m.split('\t')[1]
#     ml_dict['id'] = m.split('\t')[0]
#     ml_dict['type'] = cla_ml[claml1.index(url)][-1]
#     try:
#         title = re.findall('<title> (.*?) </title>', m)[0]
#     except:
#         continue
#     try:
#         booktitle = re.findall('<booktitle> (.*?) </booktitle>', m)[0]
#     except:
#         booktitle = 'None'
#     try:
#         category = re.findall('<type> (.*?) </type>', m)[0]
#     except:
#         category = 'None'
#     ml_dict['title'] = title
#     ml_dict['category'] = category
#     ml_dict['booktitle'] = booktitle
#     with open('cora.json', 'a', encoding='utf-8') as fw:
#         json.dump(ml_dict, fw, ensure_ascii=False)
#         fw.write('\n')
#
#
# a = [json.loads(i) for i in open('cora.json')]
# keys = ['category', 'type', 'title', 'booktitle', 'id']
#
# da = [(i[keys[0]], i[keys[1]], i[keys[2]], i[keys[3]], i[keys[4]]) for i in a]
#
# import pandas as pd
#
# daa = pd.DataFrame(da, columns=keys)
# daa = daa.drop_duplicate



cora = [(i.strip('\n').split('\t')[0], i.strip('\n').split('\t')[1]) for i in open('cora/graph.txt')]

a = pd.DataFrame(cora)
a.to_csv('cora/cora.edge', sep= ' ', header=False, index=False)

abss = [i.strip('\n') for i in open('cora/data.txt')]
abs = pd.DataFrame(abss, index=range(len(abss)))

abs.to_csv('cora/cora.map', header=False)