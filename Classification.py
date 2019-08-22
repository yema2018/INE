import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os

def SVMclassifier(emb, dim, group, test_ratio=0.3):
    da = emb
    group1 = pd.read_csv(group, sep=' ', names=['id'], index_col=[0])
    data = da.join(group1).dropna()
    data_x = data.iloc[:, 0:dim].values
    data_y = data.iloc[:, -1].astype(int).values

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio, random_state=5)

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(x_train, y_train)

    clf = SVC(gamma='auto')
    clf.fit(X_resampled, y_resampled)

    f1 = f1_score(y_test, clf.predict(x_test), average='macro')
    acc = accuracy_score(y_test, clf.predict(x_test))

    print('f1_score {:g}\t acc {:g}'.format(f1, acc))
    return f1, acc


def SVMforSemi(data, emb):
    data = data.join(emb, how='right').dropna()
    train = data[data['mask'] == 1.0]
    test = data[data['mask'] == 0.0]

    x_train = emb.loc[train.index].values
    y_train = train['label'].values
    x_test = emb.loc[test.index].values
    y_test = test['label'].values

    # x_test = pd.read_csv('dblp/temp_a0.9_oon',index_col=0).values
    # y_test = pd.read_csv('dblp/toon/oot_drop',index_col=[0])['t'].values

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(x_train, y_train)

    clf = SVC(gamma='auto')
    clf.fit(X_resampled, y_resampled)

    f1 = f1_score(y_test, clf.predict(x_test), average='macro')
    acc = accuracy_score(y_test, clf.predict(x_test))

    print('f1_score {:g}\t acc {:g}'.format(f1, acc))
    return f1, acc


if __name__ == "__main__":
    # emb = pd.read_csv('cora/cora.doc2vec',index_col=[0])
    # SVMclassifier(emb, 100, 'cora/group.txt', test_ratio=0.5)

    # fr = open('dblp/dblp_w1.emb').readlines()[1:]
    # emb = [i.strip('\n').split() for i in fr]
    # emb = pd.DataFrame(emb)
    # emb = emb.set_index([0]).astype(float)
    # emb.columns = [str(i) for i in range(100)]
    # emb.index = [int(i) for i in emb.index]
    #
    unlabeled_ratio = 0.9

    mapf = pd.read_csv('cora/cora.map', index_col=[0], names=['id'])
    labelf = pd.read_csv('cora/group.txt', sep=' ', index_col=[0], names=['label'])
    data = mapf.join(labelf).dropna()
    data['label'] = data['label'].astype(int)
    np.random.seed(5)
    shuffled = np.random.permutation(data.index)
    mask_num = int(len(data) * unlabeled_ratio)
    mask_id = shuffled[-mask_num:]
    mask = np.ones([len(data)])
    for m in range(len(data)):
        if data.index[m] in mask_id:
            mask[m] = 0
    data['mask'] = mask

    emb = pd.read_csv('cora/b10/a0.9_3_0.8658431165906076', index_col=[0])
    f1, _ = SVMforSemi(data, emb)

