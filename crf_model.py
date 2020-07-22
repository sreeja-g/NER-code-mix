from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers

import scipy.stats

import featureVec

data = featureVec.numericFeatures()

X_train = data['X_train']
y_train = data['y_train']

X_test = data['X_test']
y_test = data['y_test']


crf = CRF(algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False)


crf.fit(X_train, y_train)

labels = list(crf.classes_)
# labels.remove('O')
# labels

y_pred = crf.predict(X_test)

flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

# crf = sklearn_crfsuite.CRF(
#     algorithm='l2sgd',
#     max_iterations=100,
#     all_possible_transitions=True
# )
# params_space = {
#     'c2': scipy.stats.expon(scale=0.05),
# }

# f1_scorer = make_scorer(flat_f1_score,
#                         average='weighted', labels=labels)

# rs = RandomizedSearchCV(crf, params_space,
#                         cv=3,
#                         verbose=1,
#                         n_jobs=-1,
#                         n_iter=50,
#                         scoring=f1_scorer)
# rs.fit(X_train, y_train)

# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)
# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))