import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

X = pd.read_csv('processed_data/featureVec.csv', header=0)

y = X['word.Tag']

X.drop('word.Tag', axis=1, inplace=True)

X=X.astype('float32')
y=y.astype('float32')
X = np.nan_to_num(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = Sequential()
model.add(LSTM(100, input_shape=(32, 1)))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split = 0.2, verbose=1)

model.summary()

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)


print("Results for LSTM..")

print(classification_report(y_test, y_pred))


score = f1_score(y_pred, y_test, average='weighted')
print("LSTM F1 score: {:.2f}".format(score))
