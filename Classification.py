import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

def SVMclassifier(emb, dim, group, test_ratio=0.3):

    da = emb

    # da = pd.read_csv(emb_dir, index_col=[0])

    #net = [i.strip('\n').split() for i in open('cora/cora2.emb')][1:]
    #net = pd.DataFrame(net)
    #net = net.set_index(net[0].astype(int)).drop([0], axis=1).astype(float)
    #net.columns = range(100, 228)

    #da = da.join(net)

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




    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(data_x, data_y)
    #
    # x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.7, random_state=5)
    #
    # clf = SVC(gamma='auto')
    # clf.fit(x_train, y_train)
    #
    print('f1_score {:g}\t acc {:g}'.format(f1, acc))

    return f1, acc


if __name__ == "__main__":
    emb = pd.read_csv('dblp/sim_d200_dr0.5_h4.emb',index_col=[0])
    SVMclassifier(emb, 200, 'dblp/group.txt', test_ratio=0.3)