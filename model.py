import csv
import cv2
from keras.layers import Dense, Flatten
from keras.models import Sequential
import numpy as np

images = []
measurements = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        images.append(cv2.imread('data/IMG/' + row['center'].split('/')[-1]))
        measurements.append(float(row['steering']))

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')
