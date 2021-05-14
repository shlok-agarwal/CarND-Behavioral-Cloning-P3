import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.preprocessing.image import load_img,img_to_array

def visualize_filters(model, layer_name, channels = 3):

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Grab the filters and biases for that layer
    filters, biases = layer_dict[layer_name].get_weights()

    # Normalize filter values to a range of 0 to 1 so we can visualize them
    f_min, f_max = np.amin(filters), np.amax(filters)
    filters = (filters - f_min) / (f_max - f_min)

    # Plot first few filters
    # n_filters, index = filters.shape[3], 1 # To plot all filters
    n_filters, index = 6, 1 # To plot all filters
    for i in range(n_filters):
        f = filters[:, :, :, i]
        
        # Plot each channel separately
        for j in range(channels):

            ax = plt.subplot(n_filters, channels, index)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.imshow(f[:, :, j], cmap='viridis')
            index += 1
            
    plt.show()

def visualize_feature_map(model, layer_name, image):

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    model = Model(inputs=model.inputs, outputs=layer_dict[layer_name].output)

    # Perpare the image
    image = image.astype(float)
    image = np.expand_dims(image, axis=0)
    
    # Apply the model to the image
    feature_maps = model.predict(image)

    square = int(np.floor(np.sqrt(feature_maps.shape[3])))
    index = 1
    for _ in range(square):
        for _ in range(square):
            
            ax = plt.subplot(square, square, index)
            ax.set_xticks([])
            ax.set_yticks([])
            # print(index)
            plt.imshow(feature_maps[0, :, :, index-1], cmap='viridis')
            index += 1
            
    plt.show()