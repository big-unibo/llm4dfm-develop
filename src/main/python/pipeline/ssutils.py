from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

datasets = os.getenv('DATASETS')
outputs = os.getenv('OUTPUTS')
results = os.getenv('RESULTS')
inputs = os.getenv('INPUTS')


# Process attributes tables name (TABLE.attribute -> attribute) for matching
def preprocess_dependencies_attributes(dependency_value, keep_tables, new_names):

    # Summarize tables names if presents
    def remove_first_part(single_dependency):
        if '.' in single_dependency:
            table_name, attribute = single_dependency.split('.')[0].replace(' ', ''), single_dependency.split('.')[1]
            table_name = new_names[table_name] if table_name in new_names else table_name
            return '.'.join([table_name, attribute]) if keep_tables else attribute
        return single_dependency

    return '\n'.join([remove_first_part(word) for word in dependency_value.split(',')])


# Dependencies to consider in second step
def is_a_valid_role_dependency(dependency_key):
    return dependency_key in ['from', 'to']


# Used to store graph image
def store_image(plt, name, img_format):
    plt.savefig(f'{outputs}{Path(name).stem}.{img_format}', format=img_format)
