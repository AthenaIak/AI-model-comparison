"""
This script trains models and exports them to files.
It can be run independently of the cnn comparison model.
"""
import model_handler
import data_manipulator
from keras.datasets import mnist


def train_normal_model(train_images, train_labels):
    """
    Trains a model and exports it. All training data are used in a single fitting session.
    :param train_images:
    :param train_labels:
    :return: The trained model
    """
    normal_model = model_handler.cnn_model()

    # prepare data
    train_images, train_labels = data_manipulator.prepare_visual_data(train_images, train_labels)

    # train model
    normal_model.fit(train_images, train_labels, epochs=5, batch_size=64)

    return normal_model


def train_unbalanced_model(train_images, train_labels):
    """
    Trains a model and exports it. All training data are used in two unbalanced fitting sessions.
    :param train_images:
    :param train_labels:
    :return: The trained model
    """
    bad_model = model_handler.cnn_model()

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

    # train model
    bad_model.fit(train_images_a, train_labels_a, epochs=5, batch_size=64)
    bad_model.fit(train_images_b, train_labels_b, epochs=5, batch_size=64)

    model_handler.save_model(bad_model, 'bad_model')


# import data
(train_images, train_labels), _ = mnist.load_data()

normal_model = train_normal_model(train_images, train_labels)
model_handler.save_model(normal_model, 'normal_model')
