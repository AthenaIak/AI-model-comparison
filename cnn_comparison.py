import model_handler
import data_manipulator
from keras.datasets import mnist
import reporter

_, (test_images, test_labels) = mnist.load_data()
test_images, test_labels = data_manipulator.prepare_visual_data(test_images, test_labels)


def demo_unbalanced_model():
    normal_model = model_handler.load_model('normal_model')
    model_handler.compile_cnn_model(normal_model)
    bad_model = model_handler.load_model('bad_model')
    model_handler.compile_cnn_model(bad_model)

    # check accuracy against test data
    test_loss, test_acc_unbalanced = bad_model.evaluate(test_images, test_labels)
    print(test_acc_unbalanced)

    # check accuracy against test data
    test_loss, test_acc_normal = normal_model.evaluate(test_images, test_labels)
    print("Accuracy")
    print("Normal model", test_acc_normal)
    print("Unbalanced model", test_acc_unbalanced)

    reporter.generate_naive_report([normal_model, bad_model])


def demo_incremental_model():
    parts = 10
    incremental_models = [None] * parts
    accuracy_per_model = [None] * parts
    for part in range(parts):
        incremental_models[part] = model_handler.load_model('normal_model_part_' + str(part + 1) + '_of_' + str(parts))
        model_handler.compile_cnn_model(incremental_models[part])
        _, test_acc = incremental_models[part].evaluate(test_images, test_labels)
        accuracy_per_model[part] = test_acc
        print('Accuracy of part ' + str(part) + ': ' + str(test_acc))

    print('\nAccuracy for each part : ', accuracy_per_model)

    reporter.generate_snapshot_comparison_report(incremental_models, 'conv2d', 0, parts - 1)
    reporter.generate_snapshot_comparison_report(incremental_models, 'conv2d', 0)

    reporter.generate_snapshot_comparison_report(incremental_models, None, 0, parts - 1)
    reporter.generate_snapshot_comparison_report(incremental_models, None, 0)


demo_unbalanced_model()
demo_incremental_model()
