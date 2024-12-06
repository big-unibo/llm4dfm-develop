import os
from tests.utils import load_datasets, load_exercise, store_test_output

test_path = os.path.dirname(os.path.realpath(__file__))

preprocess_dataset_directory = os.path.join(test_path, 'dataset')

preprocess_output = os.path.join(test_path, 'generated')

def load_preprocess_datasets():
    return load_datasets(preprocess_dataset_directory)

def load_preprocess_exercise(full_name):
    return load_exercise(preprocess_dataset_directory, full_name)

def store_preprocess_test_output(outputs, ex_name):
    store_test_output(outputs, preprocess_output, ex_name)
