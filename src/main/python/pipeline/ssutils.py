from dotenv import load_dotenv
import os

load_dotenv()

datasets = os.getenv('DATASETS')
outputs = os.getenv('OUTPUTS')
results = os.getenv('RESULTS')
inputs = os.getenv('INPUTS')


# Given a different format ground-truth and model-output, uniform it
def clean_gt_dependencies(deps):
    transformed = []
    for item in deps:
        combined_dict = {}
        for sub_item in item:
            combined_dict.update(sub_item)
        transformed.append(combined_dict)
    return transformed


def remove_explicit_tables_to_output(dependency_value):
    def remove_first_part(input_string):
        if '.' in input_string:
            position = input_string.find('.')
            return input_string[position + 1:]
        return input_string

    return ''.join([remove_first_part(word) for word in dependency_value.split(' ')])


def is_a_valid_role_dependency(dependency_key):
    return dependency_key in ['from', 'to']


def store_image(plt, name, format):
    plt.savefig(f'{outputs}{name}.{format}', format=format)
