import csv
import cv2
import numpy as np

lines = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        lines.append(line)

images =[]
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

def simple(model):
    
    model.add(Flatten())
    model.add(Dense(1))
    return model

def LeNet(model):
    # ref: Conv2D( filters, kernel_size, ..)
    model.add(Conv2D(6,5,activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(6,5,activation="relu"))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

# model = simple(model)
model = LeNet(model)

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs= 7)

model.save('model.h5')