from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Lambda, Dropout, GlobalAveragePooling2D, Cropping2D, Input
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

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

def ResNet():
    
    # Makes the input placeholder layer 160,320,3
    model_input = Input(shape=(160,320,3))
    crop = Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))(model_input)
    inp = Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(65,320,3))(crop)
    resnet = ResNet50(weights=None, include_top=False, input_shape=(65,320,3))(inp)
    pool = GlobalAveragePooling2D()(resnet)
    fc1 = Dense(512, activation='relu')(pool)
    fc2 = Dense(10)(fc1)
    predictions = Dense(1)(fc2)
    # Creates the model, assuming your final layer is named "predictions"
    model = Model(inputs=model_input, outputs=predictions)
    return model
    
def GoogLeNet(model):
    # using sequential model
    model.add(InceptionV3(weights=None, include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

if __name__ == "__main__":
    # model = Sequential()
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # model = ResNet(model)
    model = ResNet2()
    print(model.summary())