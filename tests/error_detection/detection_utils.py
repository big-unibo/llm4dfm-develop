import os
from tests.utils import load_datasets, load_exercise, store_test_output

test_path = os.path.dirname(os.path.realpath(__file__))

detection_dataset_directory = os.path.join(test_path, 'dataset')

detection_output = os.path.join(test_path, 'generated')

def load_detection_datasets():
    return load_datasets(detection_dataset_directory)

def load_detection_exercise(full_name):
    return load_exercise(detection_dataset_directory, full_name)

def store_detection_test_output(outputs, ex_name):
    store_test_output(outputs, detection_output, ex_name)