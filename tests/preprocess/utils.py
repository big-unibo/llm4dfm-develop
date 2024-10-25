from pathlib import Path
import os

test_path = os.path.dirname(os.path.realpath(__file__))

metrics_dataset_directory = os.path.join(test_path, 'dataset')

metrics_output = os.path.join(test_path, 'generated')

def load_preprocess_datasets():
    file_dir = Path(metrics_dataset_directory).resolve()
    files = [f.stem for f in file_dir.iterdir() if f.is_file() and f.name != '.DS_Store']

    return files