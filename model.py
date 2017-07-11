import csv
import cv2
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

steering_correction = 0.2
batch_size=32

# load sample metadata into memory
samples = []
with open('data/driving_log.csv') as csvfile:
    for line in csv.DictReader(csvfile):
        samples.append(line)

# split sample metadata into train and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# sample generator for efficient memory usage
def generator(samples):
    while 1: # Loop forever so the generator never terminates
        # shuffle samples
        shuffle(samples)
        # batching
        for offset in range(0, len(samples), batch_size):
            images = []
            measurements = []
            for batch_sample in samples[offset:offset+batch_size]:
                # add center, left, and right images; both standard and flipped horizontally on y-axis
                for picture in ['center','left','right']:
                    image = cv2.imread('data/IMG/' + batch_sample[picture].split('/')[-1])
                    images.append(image)
                    images.append(cv2.flip(image,1))
                # add steering measurements for center (left and right include steering correction); both standard and flipped
                measurement = float(batch_sample['steering'])
                measurements.append(measurement)
                measurements.append(measurement * -1.0)
                measurements.append(measurement + steering_correction)
                measurements.append((measurement + steering_correction) * -1.0)
                measurements.append(measurement - steering_correction)
                measurements.append((measurement - steering_correction) * -1.0)

            yield shuffle(np.array(images), np.array(measurements))
            
# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# Neural network
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(36, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(48, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# model is fit using Mean Square Error loss fuction and Adam optimizer
model.compile(loss='mse', optimizer='adam')
# run up to 50 epochs, but use early termination based on validation loss knee-finding
model.fit_generator(
  train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator,
  nb_val_samples=len(validation_samples), verbose=2, nb_epoch=50, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

model.save('model.h5')
