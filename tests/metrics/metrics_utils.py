import os
from tests.utils import load_datasets, load_exercise, store_test_output

test_path = os.path.dirname(os.path.realpath(__file__))

metrics_dataset_directory = os.path.join(test_path, 'dataset_easier')

metrics_output = os.path.join(test_path, 'generated')

def load_metrics_datasets():
    return load_datasets(metrics_dataset_directory)

def load_metrics_exercise(full_name):
    return load_exercise(metrics_dataset_directory, full_name)

def store_metrics_test_output(outputs, ex_name):
    store_test_output(outputs, metrics_output, ex_name)