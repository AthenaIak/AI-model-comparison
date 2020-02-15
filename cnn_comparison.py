import ai_model_creator
import data_manipulator
from keras.datasets import mnist

normal_model = ai_model_creator.cnn_model()
bad_model = ai_model_creator.cnn_model()

# import data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## Unbalanced model
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
# prepare data
train_images, train_labels = data_manipulator.prepare_visual_data(train_images, train_labels)

# train model
normal_model.fit(train_images, train_labels, epochs=5, batch_size=64)

# check accuracy against test data
test_loss, test_acc_normal = normal_model.evaluate(test_images, test_labels)
print("Accuracy")
print("Normal model", test_acc_normal)
print("Unbalanced model", test_acc_unbalanced)
