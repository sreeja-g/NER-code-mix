from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

import csv

# import featureVec as fV
import featureVecWithFastText as fV

data = fV.numericFeatures()

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


csv_columns = ['Sentence','True','Predicted']
test_sentences = data['sentences'][int(0.7*len(data['sentences'])):]

with open('results_predicted/result.tsv', 'w') as ofile:
        writer = csv.writer(ofile, delimiter='\t')
        writer.writerow(csv_columns)

        for i in range(len(test_sentences)):

            for j in range(len(y_test[i])):

                new_row = []
                new_row.append(test_sentences[i].split(' ')[j])
                new_row.append(y_test[i][j])
                new_row.append(y_pred[i][j])

                writer.writerow(new_row) 

            writer.writerow(['','',''])  