import cv2
import numpy as np
import sklearn
import csv
# # Shuffle data for randomness
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generator(samples, batch_size=32, correct_factor = 0.3, num_cameras = 3):
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
            
                    # detect recorded data
                    tmp = source_path.split('\\' )[0]
                    if tmp == 'C:':
                        # recorded data files contain the absolute path to the image.
                        name = source_path
                    else:
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

def getDataSet(samples, augment = True, num_cameras = 3):
    images =[]
    measurements = []
    correct_factor = 0.2
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
        if augment:
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement*-1.0)

    # Neural networks
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    return X_train, y_train

def getDataGen():
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation 
        # (strictly between 0 and 1)
        validation_split=0.0)
    
    return datagen

