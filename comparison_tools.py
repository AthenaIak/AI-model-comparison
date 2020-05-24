"""
The comparison tools are functions that help compare neural networks.
"""
from scipy.stats import ks_2samp
import numpy as np


def find_layer(model, type, order=0):
    """
    Given a model, find the Nth layer of the specified type.
    :param model: the model that will be searched
    :param type:  the lowercase type, as it is automatically saved by keras in the layer's name (e.g. conv2d, dense)
    :param order: 0 by default (the first matching layer will be returned)
    :return: The index of the matching layer or None if it was not found.
    """
    num_found = 0
    for layer in model.layers:
        if type + '_' in layer.get_config()['name']:
            if order == num_found:
                return layer

            num_found += 1
    return None


def get_weights_for_layer(layer):
    """
    Extracts the weights and biases of a layer.
    :param layer: The layer that the weights and biases will be extracted from.
    :return: A triple consisting of
             a) the weights,
             b) the flattened weights (weights as a list of values),
             c) the biases.
    """
    layer_weights = layer.get_weights()[0]
    layer_weights_flat = layer_weights.reshape(layer_weights.size)
    layer_bias = layer.get_weights()[1]

    return layer_weights, layer_weights_flat, layer_bias


def get_weights_for_network(network):
    """
    Extracts the weights and biases for each layer of a network.
    :param network: The network that the layer weights and biases will be extracted from.
    :return: A triple consisting of a list of values for each layer, containing the layer's
             a) weights,
             b) flattened weights (weights as a list of values),
             c) biases.
    """
    weights_per_layer = []
    weights_flat_per_layer = []
    bias_per_layer = []
    for layer in network.layers:
        if layer.get_weights():
            layer_weights, layer_weights_flat, layer_bias = get_weights_for_layer(layer)
            weights_per_layer.append(layer_weights)
            weights_flat_per_layer.append(layer_weights_flat)
            bias_per_layer.append(layer_bias)

    return weights_per_layer, weights_flat_per_layer, bias_per_layer


def compare_distributions_with_ks(vectorA, vectorB):
    """
    Computes the Kolmogorov-Smirnov statistic on 2 vectors.
    :param vectorA: The first vector we want to compare.
    :param vectorB: The second vector we want to compare.
    :return: The Kolmogorov-Smirnov statistic.
    """
    ks = ks_2samp(vectorA, vectorB)
    return ks


def compare_distributions_with_histogram(vectorA, vectorB):
    """
    Compare the distributions of two vectors by using a histogram with identical bins.
    :param vectorA: The first vector we want to compare. The histogram bins will be determined based on this vector.
    :param vectorB: The second vector we want to compare.
    :return: A triple consisting of:
            a) the amount of datapoints the fall into each bin for vectorA
            b) the amount of datapoints the fall into each bin for vectorB
            c) the bins
    """
    hist_a, bins = np.histogram(vectorA)
    hist_b, _ = np.histogram(vectorB, bins)
    return hist_a, hist_b, bins


def calculate_layer_distance(ref_weights, comp_weights):
    """
    Calculates the distance between two layers. The layers should have the same shape.
    It is assumed that the layers belong to different snapshots of the same model,
    and there exists a 1-to-1 correspondence between neurons of the two layers.
    :param ref_weights:  The weights of the reference layer
    :param comp_weights: The weights of the layer that is compared to the reference one
    :return: A tuple consisting of:
            a) the mean squared error per filter layer
            b) the mean root squared error per filter in the layer
    """
    se = np.square(ref_weights - comp_weights)
    mse = se.mean(axis=tuple(range(0, se.ndim - 1)))
    rmse = np.sqrt(mse)
    return mse, rmse


def metric_to_csv(metric):
    """
    Exports a csv from an list of metrics.
    :param metric: The comparison metric.
                   The first dimension refers to the layers of the network.
                   The second to the combinations of networks that were compared.
                   The third is the metric calculated for each layer's filter.
    :return: The list is printed in the console in a csv format.
    """
    num_layers = len(metric)
    num_combos = len(metric[0])
    for layer_num in range(num_layers):
        idx_sorted_filters = np.argsort(np.mean(metric[layer_num][:], axis=0)[::-1])
        print("Layer,Model,", end="")
        for filter_idx in idx_sorted_filters:
            print("Filter", filter_idx, end=",")
        print()
        for combo in range(num_combos):
            print('%d,%d,%s' % (layer_num, combo, ','.join(map(str, metric[layer_num][combo][idx_sorted_filters]))))
