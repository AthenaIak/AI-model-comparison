import comparison_tools
from itertools import combinations


def generate_naive_report(models):
    for amodel, bmodel in combinations(models, 2):
        a_dense_layer = comparison_tools.find_layer(amodel, 'dense')
        _, a_dense_weights, a_dense_biases = comparison_tools.get_weights_for_layer(a_dense_layer)

        b_dense_layer = comparison_tools.find_layer(bmodel, 'dense')
        _, b_dense_weights, b_dense_biases = comparison_tools.get_weights_for_layer(b_dense_layer)

        print('Comparison between weights:')
        print('K-S comparison (the distribution is similar when the K-S statistic is small and p > 0.05):')
        weights_ks = comparison_tools.compare_distributions_with_ks(a_dense_weights, b_dense_weights)
        print('K-S=%.6s p=%.5s' % weights_ks)

        print('Histogram comparison')
        a_weights_hist, b_weights_hist, weights_bins = comparison_tools.compare_distributions_with_histogram(
            a_dense_weights,
            b_dense_weights)
        print('Model A: %s\nModel B: %s' % (a_weights_hist, b_weights_hist))
        print('Bins   : %s\n' % weights_bins)

        print('Comparison between biases:')
        print('K-S comparison (the distribution is similar when the K-S statistic is small and p > 0.05):')
        biases_ks = comparison_tools.compare_distributions_with_ks(a_dense_biases, b_dense_biases)
        print('K-S=%.6s p=%.5s' % biases_ks)

        print('Histogram comparison')
        a_biases_hist, b_biases_hist, biases_bins = comparison_tools.compare_distributions_with_histogram(
            a_dense_biases,
            b_dense_biases)
        print('Model A: %s\nModel B: %s' % (a_biases_hist, b_biases_hist))
        print('Bins   : %s\n' % biases_bins)


def generate_snapshot_comparison_report(models, layer_type, layer_order=0, ref_idx=None):
    """
    Outputs the results of the comparison between multiple models.
    The models are compared to a single reference model, if one is supplied.
    Otherwise, each model is compared to the next one in the list.
    Only one layer is compared between the models.
    It is assumed that the models supplied are snapshots/evolutions of the same model, as
    and there exists a 1-to-1 correspondence between neurons of the models.
    :param models:      A list of models.
    :param layer_type:  The lowercase type, as it is automatically saved by keras in the layer's name (e.g. conv2d, dense)
    :param layer_order: 0 by default (the first matching layer will be returned)
    :param ref_idx:     The index of the reference model. None by default (models are compared to the next one in the list)
    :return: A report in the console. No values returned.
    """
    # calculate the total number of models supplied
    parts = len(models)

    # make a list containing the weights of the examined layer for each model
    layer_weights = [None] * parts
    for part in range(parts):
        examined_layer = comparison_tools.find_layer(models[part], layer_type, layer_order)
        layer_weights[part], _, _ = comparison_tools.get_weights_for_layer(examined_layer)

    if ref_idx is None:
        print('\nRoot mean squared error between a model and the next one in the sequence: ')
        for part in range(parts - 1):
            _, rmse = comparison_tools.calculate_layer_distance(layer_weights[part], layer_weights[part + 1])
            print('part %3d vs part %3d: %f\n%s' % (part + 1, part + 2, rmse.mean(), rmse))
    else:
        # make a list of the indexes of the models that should be compared to the reference one
        compared_idxs = list(filter(lambda a: a != ref_idx, [idx for idx in range(parts)]))

        print('\nRoot mean squared error between the reference model and other models: ')
        for part in compared_idxs:
            _, rmse = comparison_tools.calculate_layer_distance(layer_weights[ref_idx], layer_weights[part])
            print('part %3d: %f\n%s' % (part + 1, rmse.mean(), rmse))
