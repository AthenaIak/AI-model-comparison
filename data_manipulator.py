import numpy as np
from keras.utils import to_categorical

'''
Convert images to 28x28 with decimal values and labels to categorical
'''


def prepare_visual_data(images, labels):
    images = images.reshape((len(images), 28, 28, 1))
    images = images.astype('float32') / 255
    labels = to_categorical(labels)
    return images, labels


'''
Split data to two mutually-exclusive unbalanced sets. 
The first set is returned (the second can be inferred as returned_val == False)
args:
  - labels        : the labels of the whole set that should be split
  - major_labels  : the label values that the returned set should favour
  - minor_labels  : the label values that the returned set should not favour
  - bleed_through : the number of minor elements that bleed through to the returned set 
                    (expected to be < sum / 2 of each category)
returns:
  - a boolean numpy array that can be used to access the labels/data for the first unbalanced set
'''


def split_unbalanced(labels, major_categories, minor_categories, bleed_through):
    #### Split training data to two unbalanced sets
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
