from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Lambda, Dropout, GlobalAveragePooling2D, Cropping2D, Input, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from resnet_v1 import resnet_v1
import tensorflow as tf
from visualize_layers import visualize_filters, visualize_feature_map
import matplotlib.image as mpimg

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
    # Makes the input placeholder layer 160,320,3
    model_input = Input(shape=(160,320,3))
    crop = Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))(model_input)
    inp = Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(65,320,3))(crop)

    c1 = Conv2D(24,5, strides=(2,2), activation="relu")(crop)
    c2 = Conv2D(36,5, strides=(2,2), activation="relu")(c1)
    c3 = Conv2D(48,5, strides=(2,2))(c2)
    bnorm = BatchNormalization()(c3)
    act = Activation('relu')(bnorm)
    c4 = Conv2D(64,3, activation="relu")(act)
    c5 = Conv2D(64,3, activation="relu")(c4)

    f = Flatten()(c5)
    fc1 = Dense(100)(f)
    drop = Dropout(0.5)(fc1)
    fc2 = Dense(50)(drop)
    fc3 = Dense(10)(fc2)
    predictions = Dense(1)(fc3)
    model = Model(inputs=model_input, outputs=predictions)
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

def Resnet_18():
    return resnet_v1(input_shape=(160, 320, 3), depth=20)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 50:
        lr *= 1e-3
    elif epoch > 30:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def GoogLeNet():
    # using sequential model
    # Makes the input placeholder layer 160,320,3
    model_input = Input(shape=(160,320,3))
    crop = Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))(model_input)
    # Re-sizes the input with Kera's Lambda layer & attach to crop layer
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
    model = Nvidia()
    print(model.summary())

    # viz stuff
    # visualize_filters(model, 'conv2d_2')
    # current_path = 'data/data/IMG/' + 'center_2016_12_01_13_33_07_834.jpg'
    # image = mpimg.imread(current_path)
    # visualize_feature_map(model, 'conv2d_2', image)