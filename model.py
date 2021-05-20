import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from model_lib import simple, LeNet, GoogLeNet, Resnet_18, Resnet_50, Nvidia, lr_schedule
from preprocess_data import generator, getDataSet, getDataGen
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

samples = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        samples.append(line)

# recorded data
with open('data/recorded_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
BATCH_SIZE= 64
num_epochs = 20

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = Nvidia()
print(model.summary())

model.compile(loss='mse', optimizer='adam')
# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

history_object = model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/BATCH_SIZE), 
            epochs=num_epochs, verbose=1, callbacks = callbacks)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("TrainingHistory")
plt.show()