import re
from pathlib import Path
import yaml

def load_files():

    # Get the current working directory (which is 'tests/metrics')
    current_directory = Path.cwd()

    # Define the path to the 'dataset' folder
    dataset_directory = current_directory / 'dataset'

    # List all files with full paths in the 'dataset' directory
    files = [f.stem for f in dataset_directory.iterdir() if f.is_file() and f.name != '.DS_Store']

    return files

def get_info_from_filename(filename):
    fn = filename.split('-')
    exercise = '-'.join(fn[:2])
    version = fn[2]

    filename_no_data = re.sub(r'-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$', '', filename)
    prompt = '-'.join(filename_no_data.split('-')[3:])

    return exercise, version, prompt


def get_output_from_filename(filename):
    with open(f'{Path().absolute()}/../dataset/{filename}.yml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
