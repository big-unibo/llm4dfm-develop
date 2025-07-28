import os
from tests.utils import load_datasets, store_test_output, load_exercise

test_path = os.path.dirname(os.path.realpath(__file__))

parsable_dataset_directory = os.path.join(test_path, 'dataset_parsable')
unparsable_dataset_directory = os.path.join(test_path, 'dataset_unparsable')

parsing_output = os.path.join(test_path, 'generated')

def load_unparsable_datasets():
    return load_datasets(unparsable_dataset_directory)

def load_parsable_datasets():
    return load_datasets(parsable_dataset_directory)

def load_parsable_exercise(full_name):
    return load_exercise(parsable_dataset_directory, full_name)

def load_unparsable_exercise(full_name):
    return load_exercise(unparsable_dataset_directory, full_name)

def store_parsing_test_output(outputs, ex_name):
    store_test_output(outputs, parsing_output, ex_name)
