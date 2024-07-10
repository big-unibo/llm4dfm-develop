from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

datasets = os.getenv('DATASETS')
outputs = os.getenv('OUTPUTS')
results = os.getenv('RESULTS')
inputs = os.getenv('INPUTS')


# Process attributes tables name (TABLE.attribute -> attribute) for matching
def preprocess_dependencies_attributes(dependency_value):

    # Summarize tables names if presents
    def remove_first_part(input_string):
        if '.' in input_string:
            table_name, attribute = input_string.split('.')[0], input_string.split('.')[1]
            # Pick only 2 char for each table name part (i.e. RACING_STABLES -> RA_ST)
            table_name = '_'.join([tb_n[:2] for tb_n in table_name.split('_')])
            return '.'.join([table_name, attribute])
        return input_string

    return '\n'.join([remove_first_part(word) for word in dependency_value.split(',')])


# Dependencies to consider in second step
def is_a_valid_role_dependency(dependency_key):
    return dependency_key in ['from', 'to']


# Used to store graph image
def store_image(plt, name, img_format):
    plt.savefig(f'{outputs}{Path(name).stem}.{img_format}', format=img_format)
