import ai_model_creator
import data_manipulator
import comparison_tools
from keras.datasets import mnist

# import data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## Unbalanced model
bad_model = ai_model_creator.cnn_model()

# split the train images and labels to unbalanced sub-sets
train_half_set_iter = data_manipulator.split_unbalanced(train_labels, [i for i in range(0, 5)],
                                                        [i for i in range(5, 10)], 100)
train_images_a = train_images[train_half_set_iter]
train_images_b = train_images[train_half_set_iter == False]
train_labels_a = train_labels[train_half_set_iter]
train_labels_b = train_labels[train_half_set_iter == False]

# prepare data
train_images_a, train_labels_a = data_manipulator.prepare_visual_data(train_images_a, train_labels_a)
train_images_b, train_labels_b = data_manipulator.prepare_visual_data(train_images_b, train_labels_b)
test_images, test_labels = data_manipulator.prepare_visual_data(test_images, test_labels)

# train model
bad_model.fit(train_images_a, train_labels_a, epochs=5, batch_size=64)
bad_model.fit(train_images_b, train_labels_b, epochs=5, batch_size=64)

# check accuracy against test data
test_loss, test_acc_unbalanced = bad_model.evaluate(test_images, test_labels)
print(test_acc_unbalanced)

## Normal model
normal_model = ai_model_creator.cnn_model()

# prepare data
train_images, train_labels = data_manipulator.prepare_visual_data(train_images, train_labels)

# train model
normal_model.fit(train_images, train_labels, epochs=5, batch_size=64)

# check accuracy against test data
test_loss, test_acc_normal = normal_model.evaluate(test_images, test_labels)
print("Accuracy")
print("Normal model", test_acc_normal)
print("Unbalanced model", test_acc_unbalanced)

nm_dense_layer = comparison_tools.find_layer(normal_model, 'dense')
_, nm_dense_weights, nm_dense_biases = comparison_tools.get_weights_for_layer(nm_dense_layer)

bm_dense_layer = comparison_tools.find_layer(bad_model, 'dense')
_, bm_dense_weights, bm_dense_biases = comparison_tools.get_weights_for_layer(bm_dense_layer)

print('Comparison between weights:')
print('K-S comparison (the distribution is similar when the K-S statistic is small and p > 0.05):')
weights_ks = comparison_tools.compare_distributions_with_ks(nm_dense_weights, bm_dense_weights)
print('K-S=%.6s p=%.5s' % weights_ks)

print('Histogram comparison')
nm_weights_hist, bm_weights_hist, weights_bins = comparison_tools.compare_distributions_with_histogram(nm_dense_weights,
                                                                                                       bm_dense_weights)
print('normal: %s\n   bad: %s' % (nm_weights_hist, bm_weights_hist))
print('bins  : %s\n' % weights_bins)

print('Comparison between biases:')
print('K-S comparison (the distribution is similar when the K-S statistic is small and p > 0.05):')
biases_ks = comparison_tools.compare_distributions_with_ks(nm_dense_biases, bm_dense_biases)
print('K-S=%.6s p=%.5s' % biases_ks)

print('Histogram comparison')
nm_biases_hist, bm_biases_hist, biases_bins = comparison_tools.compare_distributions_with_histogram(nm_dense_biases,
                                                                                                    bm_dense_biases)
print('normal: %s\n   bad: %s' % (nm_biases_hist, bm_biases_hist))
print('bins  : %s\n' % bm_biases_hist)
