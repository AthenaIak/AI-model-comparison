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
