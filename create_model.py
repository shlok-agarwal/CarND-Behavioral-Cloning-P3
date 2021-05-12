import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from models import *
from preprocess_data import *
from resnet_v1 import resnet_v1

USE_RESNET18 = False

samples = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size = 32
epochs = 5

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# model = simple()
# model = LeNet()
# model = Nvidia()
# model = GoogLeNet(model)

# model = ResNet2()
# model = build_ResNet('ResNet18', 1)

# print(model.summary())
model = resnet_v1(input_shape=(160, 320, 3), depth=20)
model.compile(loss='mse', optimizer='adam')

X_train, y_train = getDataSet(samples)
num_examples = len(X_train)

print("Training...")
print()
for i in range(epochs):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        history_object = model.fit(batch_x, batch_y, validation_split=0.2, shuffle=True, epochs= 1) 

# history_object = model.fit_generator(train_generator, 
#             steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
#             validation_data=validation_generator, 
#             validation_steps=np.ceil(len(validation_samples)/batch_size), 
#             epochs=5, verbose=1)

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