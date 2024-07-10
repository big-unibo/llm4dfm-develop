from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

datasets = os.getenv('DATASETS')
outputs = os.getenv('OUTPUTS')
results = os.getenv('RESULTS')
inputs = os.getenv('INPUTS')

# TODO fix this
# Remove in output eventual explicit tables name (TABLE.attribute -> attribute) for matching
def remove_explicit_tables_to_output(dependency_value):
    def remove_first_part(input_string):
        if '.' in input_string:
            position = input_string.find('.')
            return input_string[position + 1:]
        return input_string

    return ''.join([remove_first_part(word) for word in dependency_value.split(' ')])


# Dependencies to consider in second step
def is_a_valid_role_dependency(dependency_key):
    return dependency_key in ['from', 'to']


# Used to store graph image
def store_image(plt, name, img_format):
    plt.savefig(f'{outputs}{Path(name).stem}.{img_format}', format=img_format)
