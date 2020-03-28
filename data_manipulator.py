"""
The data manipulator contains functions to change the format of data.
"""
import numpy as np
from keras.utils import to_categorical


def prepare_visual_data(images, labels):
    """
    Convert images to 28x28 with decimal values. Convert labels to categorical.
    :param images: Images provided as a list of vectors (size Nx784) with values 0-255.
    :param labels: Labels provided as a list of numbers.
    :return: Reshaped images (shape=28x28 and values=0-1) and labels (converted to a binary class matrix).
    """
    images = images.reshape((len(images), 28, 28, 1))
    images = images.astype('float32') / 255
    labels = to_categorical(labels)
    return images, labels


def split_unbalanced(labels, major_categories, minor_categories, bleed_through):
    """
    Split data to two mutually-exclusive unbalanced sets.
    The first set is returned (the second can be inferred as returned_val == False)
    :param labels:           the labels of the whole set that should be split
    :param major_categories: the label values that the returned set should favour
    :param minor_categories: the label values that the returned set should not favour
    :param bleed_through:    the number of minor elements that bleed through to the returned set
                             (expected to be < sum / 2 of each category)
    :return: a boolean numpy array that can be used to access the labels/data for the first unbalanced set
    """

    # get the indexes per category (0-9)
    all_categories = major_categories + minor_categories
    idx_per_category = [i for i in range(len(all_categories))]
    for i in all_categories:
        idx_per_category[i] = [idx for idx, val in enumerate(labels) if val == i]

    subset_idx = []
    for major_category in major_categories:
        subset_idx.extend(idx_per_category[major_category][bleed_through:])
    for minor_category in minor_categories:
        subset_idx.extend(idx_per_category[minor_category][:bleed_through])

    # convert the indexes to a boolean array so that we can more easily access all elements in other arrays
    set_identifier = np.zeros((len(labels)), dtype=bool)
    for idx in subset_idx:
        set_identifier[idx] = True

    return set_identifier
