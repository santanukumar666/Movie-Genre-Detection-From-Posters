import sys
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

import preprocessing as pp

num_of_filters = 32
num_of_labels = pp.n_classes
batch_size = 4
epochs = 24

print(num_of_labels)
print(pp.x_train.shape)
model = Sequential()

# 1 and 2 layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
          activation='relu', input_shape=(pp.x_train.shape[1:])))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# 3 and 4 layer
model.add(Conv2D(filters=64, kernel_size=(3, 3),
          padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_of_labels, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(pp.x_train, pp.y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(pp.x_test, pp.y_test), shuffle=True)

# saving model
gpre_json = model.to_json()
with open("gpre.json", "w") as json_file:
    json_file.write(gpre_json)
model.save_weights("gpre.h5")
