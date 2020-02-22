from scipy.stats import ks_2samp
import numpy as np

'''
Find the Nth layer of the specified type.
args:
- model : the model that will be searched
- type  : the lowercase type, as it is automatically saved by keras in the layer's name (e.g. conv2d, dense)
- order : 0 by default (the first matching layer will be returned)
'''


def find_layer(model, type, order=0):
    num_found = 0
    for layer in model.layers:
        if type + '_' in layer.get_config()['name']:
            if order == num_found:
                return layer

            num_found += 1
    return None


def get_weights_for_layer(layer):
    layer_weights = layer.get_weights()[0]
    layer_weights_flat = layer_weights.reshape(layer_weights.size)
    layer_bias = layer.get_weights()[1]

    return layer_weights, layer_weights_flat, layer_bias


def compare_distributions_with_ks(vectorA, vectorB):
    ks = ks_2samp(vectorA, vectorB)
    return ks


def compare_distributions_with_histogram(vectorA, vectorB):
    hist_a, bins = np.histogram(vectorA)
    hist_b, _ = np.histogram(vectorB, bins)
    return hist_a, hist_b, bins
