from pathlib import Path
import os
import re
from llm4dfm.pipeline.utils import load_yaml, write_yaml

test_path = os.path.dirname(os.path.realpath(__file__))

preprocess_dataset_directory = os.path.join(test_path, 'dataset')

preprocess_output = os.path.join(test_path, 'generated')

def load_preprocess_datasets():
    file_dir = Path(preprocess_dataset_directory).resolve()
    files = [f.stem for f in file_dir.iterdir() if f.is_file() and f.name != '.DS_Store']

    return files


def get_info_from_filename(filename):
    fn = filename.split('-')
    exercise = '-'.join(fn[:2])
    version = fn[2]

    filename_no_data = re.sub(r'-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$', '', filename)
    prompt = '-'.join(filename_no_data.split('-')[3:])

    return exercise, version, prompt

def load_exercise(full_name):
    return load_yaml(f'{preprocess_dataset_directory}/{full_name}')

def store_test_output(outputs, ex_name):
    os.makedirs(f'{preprocess_output}', exist_ok=True)
    write_yaml(f'{preprocess_output}/{ex_name}', outputs)
