import re
from pathlib import Path
import os
from llm4dfm.pipeline.utils import load_yaml, write_yaml

def load_datasets(dataset_directory):
    file_dir = Path(dataset_directory).resolve()
    files = [f.stem for f in file_dir.iterdir() if f.is_file() and f.suffix == '.yml']

    return files

def get_info_from_filename(filename):
    fn = filename.split('-')
    exercise = '-'.join(fn[:2])
    version = fn[2]

    filename_no_data = re.sub(r'-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$', '', filename)
    prompt = '-'.join(filename_no_data.split('-')[3:])

    return exercise, version, prompt


def load_exercise(dataset_directory, full_name):
    return load_yaml(f'{dataset_directory}/{full_name}')


def store_test_output(outputs, outputs_dir, ex_name):
    os.makedirs(f'{outputs_dir}', exist_ok=True)
    write_yaml(f'{outputs_dir}/{ex_name}', outputs)
