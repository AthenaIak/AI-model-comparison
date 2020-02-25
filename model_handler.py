from keras import layers
from keras import models


def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    compile_cnn_model(model)
    return model


def compile_cnn_model(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def save_model(model, name):
    model_out = model.to_yaml()
    with open('saved_models/' + name + '.yaml', 'w') as model_file:
        model_file.write(model_out)

    model.save_weights('saved_models/' + name + '.h5')


def load_model(name):
    model_file = open('saved_models/' + name + '.yaml', 'r')
    loaded_model_yaml = model_file.read()
    model_file.close()

    loaded_model = models.model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights('saved_models/' + name + '.h5')

    return loaded_model
