import matplotlib.pyplot as plt
import numpy as np

def visualize_filters(model, layer_name):

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
        for j in range(3):

            ax = plt.subplot(n_filters, 3, index)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.imshow(f[:, :, j], cmap='viridis')
            index += 1
            
    plt.show()