import csv
import cv2
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

import numpy as np

images = []
measurements = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image = cv2.imread('data/IMG/' + row['center'].split('/')[-1])
        images.append(image)
        images.append(cv2.flip(image,1))
        measurement = float(row['steering'])
        measurements.append(measurement)
        measurements.append(measurement * -1.0)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

model.save('model.h5')
