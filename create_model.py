import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from models import simple, LeNet, GoogLeNet, Resnet_18, Resnet_50, Nvidia
from preprocess_data import generator

samples = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size=32
num_epochs = 5

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Nvidia()
print(model.summary())

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/batch_size), 
            epochs=num_epochs, verbose=1)

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
plt.show()