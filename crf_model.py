from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

import featureVec
# import feactureVecWithLang

data = featureVec.numericFeatures()
# data = feactureVecWithLang.numericFeatures()

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

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

print(flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
)