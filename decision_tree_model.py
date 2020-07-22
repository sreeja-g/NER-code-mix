import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import plot_funcs

X = pd.read_csv('processed_data/featureVec.csv')

y = X['word.Tag']

X.drop('word.Tag', axis=1, inplace=True)

X=X.astype('float32')
y=y.astype('float32')
X = np.nan_to_num(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

dtc = DecisionTreeClassifier(max_depth=32, class_weight={0:1, 1:50, 2:18, 3:1940, 4:70, 5:3, 6:25})
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
target_names = ['I-PLACE', 'B-ORGANISATION', 'I-PERSON', 'O', 'B-PERSON', 'I-ORGANISATION', 'B-PLACE']

print("Results for Decision tree..")

print(classification_report(y_test, y_pred, target_names=target_names))


score = f1_score(y_pred, y_test, average='weighted')
print("Decision Tree F1 score: {:.2f}".format(score))
