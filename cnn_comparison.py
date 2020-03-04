import model_handler
import data_manipulator
from keras.datasets import mnist
import reporter

_, (test_images, test_labels) = mnist.load_data()
test_images, test_labels = data_manipulator.prepare_visual_data(test_images, test_labels)

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
