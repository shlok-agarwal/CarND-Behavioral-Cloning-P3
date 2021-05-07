from keras.layers import Flatten, Dense, Conv2D, MaxPool2D

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

def Nvidia(model):
    model.add(Conv2D(24,5, strides=(2,2), activation="relu"))
    model.add(Conv2D(36,5, strides=(2,2), activation="relu"))
    model.add(Conv2D(48,5, strides=(2,2), activation="relu"))
    model.add(Conv2D(64,3, activation="relu"))
    model.add(Conv2D(64,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model