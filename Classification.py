import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

da = pd.read_csv('cora/cora_net.emb', index_col=[0])

group = [i.strip('\n') for i in open('cora/group.txt')]
group1 = pd.DataFrame(group)

data = da.join(group1)

data_x = data.iloc[:, 0:100].values
data_y = data.iloc[:, -1].astype(int).values

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=5)

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(x_train, y_train)

clf = SVC(gamma='auto')
clf.fit(X_resampled, y_resampled)

print(f1_score(y_test, clf.predict(x_test), average='macro'))


# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_sample(data_x, data_y)
#
# x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=5)
#
# clf = SVC(gamma='auto')
# clf.fit(x_train, y_train)
#
# print(accuracy_score(y_test, clf.predict(x_test)))