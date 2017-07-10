import csv
import cv2
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np

steering_correction = 1.0
batch_size=32


samples = []
with open('data/driving_log.csv') as csvfile:
    for line in csv.DictReader(csvfile):
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples):
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            images = []
            measurements = []
            for batch_sample in samples[offset:offset+batch_size]:
                for picture in ['center','left','right']:
                    image = cv2.imread('data/IMG/' + batch_sample[picture].split('/')[-1])
                    images.append(image)
                    images.append(cv2.flip(image,1))
                measurement = float(batch_sample['steering'])
                measurements.append(measurement)
                measurements.append(measurement * -1.0)
                measurements.append(measurement + steering_correction)
                measurements.append((measurement + steering_correction) * -1.0)
                measurements.append(measurement - steering_correction)
                measurements.append((measurement - steering_correction) * -1.0)

            # trim image to only see section with road
            yield shuffle(np.array(images), np.array(measurements))
            
# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

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
model.fit_generator(
  train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator,
  nb_val_samples=len(validation_samples), verbose=2, nb_epoch=50, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

model.save('model.h5')
