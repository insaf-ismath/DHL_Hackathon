import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

lookup = {
    'M':0,
    'B':1
}

df = pd.read_csv('wdbc.csv')
data_set = df.values

X = data_set[:,2::]
Y_temp = data_set[:,1]
Y = []
for item in Y_temp:
    Y.append(lookup[item])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = svm.SVC()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

example_measures = np.array([[13.45,18.3,86.6,555.1,0.1022,0.08165,0.03974,0.0278,0.1638,0.0571,0.295,1.373,2.099,25.22,0.005884,0.01491,0.01872,0.009366,0.01884,0.001817,15.1,25.94,97.59,699.4,0.1339,0.1751,0.1381,0.07911,0.2678,0.06603]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)