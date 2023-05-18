import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import joblib
from sklearn import datasets
data_size = 40
folder_path = 'HandPose'
files = [cls.replace('.txt', '') \
    for cls in os.listdir(path=folder_path) \
        if 'NONE' not in cls]
X = []
Y = [cls for cls in files for _ in range(data_size)]
for file in files:
    tmp = []
    with open(f'./{folder_path}/{file}.txt', 'r') as f:
        line = f.readline()
        while line:
            line = line.strip().split(', ')
            # line = [int(line[i][j]) - int(line[0][j]) for i in range(63) for j in range(3)]
            line = [int(i) for i in line]
            tmp.append(line)
            line = f.readline()
    # tmp = np.array(tmp, dtype=int)
    X.extend(tmp)
# print(X)
X = np.array(X, dtype=int)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.2, random_state=42)
    
    
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc)

joblib.dump(clf, 'svm_clf.pkl')


class HandPoseRecognizer:
    def __init__(self):
        self.clf = joblib.load('svm_clf.pkl')
        
    def inf(self, sample):
        return self.clf(sample)