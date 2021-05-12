from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Lambda, Dropout, GlobalAveragePooling2D, Cropping2D, Input
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from resnet_v1 import resnet_v1
import tensorflow as tf

def simple():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def LeNet():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
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

def Nvidia():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
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

def Resnet_50():
    
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

def ResNet_18():
    return resnet_v1(input_shape=(160, 320, 3), depth=20)

def GoogLeNet():
    # using sequential model
    # Makes the input placeholder layer 160,320,3
    model_input = Input(shape=(160,320,3))
    crop = Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))(model_input)
    # Re-sizes the input with Kera's Lambda layer & attach to cifar_input
    resized_input = Lambda(lambda image: tf.image.resize(image, (139, 210)))(crop)
    inp = Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(65,320,3))(resized_input)
    inception = InceptionV3(weights=None, include_top=False)(model_input) #(inp)
    pool = GlobalAveragePooling2D()(inception)
    fc1 = Dense(512)(pool)
    fc2 = Dense(10)(fc1)
    predictions = Dense(1)(fc2)
    # Creates the model, assuming your final layer is named "predictions"
    model = Model(inputs=model_input, outputs=predictions)
    return model

if __name__ == "__main__":
    model = LeNet()
    print(model.summary())