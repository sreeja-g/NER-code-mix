import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dataset = read_csv('processed_data/featureVec.csv', header=0)
val = dataset.values
val=val.astype('float32')
val = np.nan_to_num(val)

X = val[:,:32]
Y = val[:,32]


X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(100, input_shape=(32, 1)))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax')) #7 class classification.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(X, Y, epochs=5, batch_size=32, validation_split = 0.2, verbose=1)

model.summary()