from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Lambda, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50

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

def ResNet(model):
    # using model API for reference
    # resnet = ResNet50(weights=None, include_top=False, input_shape=(65,320,3))
    # # adding top layers
    # x = resnet.output
    # x = GlobalAveragePooling2D()(x) # pooling
    # x = Dropout(0.7)(x) # dropout
    # predictions = Dense(1)(x)
    # model = Model(inputs = resnet.input, outputs = predictions)

    # using sequential model
    model.add(ResNet50(weights=None, include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.7))
    model.add(Dense(1))
    return model

if __name__ == "__main__":
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model = ResNet(model)
    print(model.summary())