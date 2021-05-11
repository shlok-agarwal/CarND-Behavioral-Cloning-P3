import cv2
import numpy as np
import sklearn
import csv
# # Shuffle data for randomness
from sklearn.utils import shuffle

def generator(samples, batch_size=32, correct_factor = 0.2, num_cameras = 3):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(num_cameras):

                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    name = 'data/data/IMG/' + filename
                    
                    image = cv2.imread(name)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    if i == 1: # left camera, steer right
                        measurement += correct_factor
                    elif i == 2: # right camera, steer left
                        measurement -= correct_factor
                    measurements.append(measurement)
            
            # augment data  by flipping images
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def getDataSet(samples):
    images =[]
    measurements = []
    correct_factor = 0.2
    num_cameras = 3
    for line in samples:
        for i in range(num_cameras):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = 'data/data/IMG/' + filename
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            if i == 1: # left camera, steer right
                measurement += correct_factor
            elif i == 2: # right camera, steer left
                measurement -= correct_factor
            measurements.append(measurement)

    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*-1.0)

    # Neural networks
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    return X_train, y_train
