from pathlib import Path
from dotenv import load_dotenv
import os
import collections
import yaml

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


# Load edges from dependency set filtering for valid role dependency
def load_edges(dependency_set):
    return set(
        frozenset((key, get_clean_table_attribute(value))
                  for key, value in d.items() if is_a_valid_role_dependency(key))
        for d in dependency_set if is_a_valid_dependency(d))


# Load nodes cleaned from edges set
def load_nodes(edges_set):
    return set(
        get_clean_table_attribute(entry[1])
        for fr_set in edges_set
        for entry in fr_set
    )


# Dependencies to consider in second step
def is_a_valid_role_dependency(dependency_key):
    return dependency_key in ['from', 'to']


# Filter dict of ground truth if created
def is_a_valid_dependency(dependency_dict):
    if 'refinements' in dependency_dict:
        return dependency_dict['refinements'] != 'created'
    return True


# Turns a table attribute to first letter capitalized to obtain a uniform comparison between gt and output
def get_clean_table_attribute(table_attr):
    if ',' in table_attr:
        new_val = ''
        for attrs in table_attr.split(','):
            if '.' in attrs:
                attr_split = attrs.split('.')
                val = attr_split[0] + '.' + attr_split[1][0].upper() + attr_split[1][1:]
            else:
                val = attrs.capitalize()
            if new_val == '':
                new_val += val
            else:
                new_val += ',' + val
    else:
        if '.' in table_attr:
            attr_split = table_attr.split('.')
            new_val = attr_split[0] + '.' + attr_split[1][0].upper() + attr_split[1][1:]
        else:
            new_val = table_attr[0].upper() + table_attr[1:]
    return new_val.replace(' ', '')


# Given gt and output as set, return short table names composed of 2 letters and a digit only if 2 letters appears in
# more than 1 table name
# e.g. if there is SUPPLY and SUPPLIER -> SU and SU1
def short_names_from_tables(gt, output):
    # Extract tables name to obtain short names

    all_tables_gt = set(
        val.split('.')[0].replace(' ', '')
        for subset in gt
        for _, value in subset
        for val in value.split(',')
        if '.' in val
    )
    all_tables_out = set(
        val.split('.')[0].replace(' ', '')
        for subset in output
        for _, value in subset
        for val in value.split(',')
        if '.' in val
    )

    short_names = dict()
    # Initialize a set to keep track of the used two-letter values
    used_names = set()

    # Iterate over each value in the set
    for table in all_tables_gt.union(all_tables_out):
        # Get the first two letters of the value
        new_name = '_'.join([short[:2] for short in table.split('_')])
        i = 0
        inserted = False
        while not inserted:
            if new_name not in used_names:
                inserted = True
                short_names[table] = new_name
                used_names.add(new_name)
            else:
                if i > 0:
                    new_name = new_name[:-len(str(i))]
                new_name = new_name + str(i)
                i += 1
    return short_names


# Calculates metrics from ground_truth set and generated set
def get_metrics(ground_truth, generated):
    tp = ground_truth & generated
    fn = ground_truth - tp
    fp = generated - tp

    tp_count = len(tp)
    fn_count = len(fn)
    fp_count = len(fp)

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

    return precision, recall, f1


# Build list of edges of type from -> to
def get_tp_fn_fp_edges_to_list(ground_truth, generated):
    tp = ground_truth & generated
    fn = ground_truth - tp
    fp = generated - tp
    tp_list = [collections.OrderedDict(sorted(fs)) for fs in tp]
    tp_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))
    fn_list = [collections.OrderedDict(sorted(fs)) for fs in fn]
    fn_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))
    fp_list = [collections.OrderedDict(sorted(fs)) for fs in fp]
    fp_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))

    return tp_list, fn_list, fp_list


# Used to store graph image
def store_image(plt, name, img_format):
    plt.savefig(f'{outputs}{Path(name).stem}.{img_format}', format=img_format)


def update_output_with_metrics(file, result_with_metrics):
    with open(f'{outputs}{file}', 'w+', encoding='utf-8') as outfile:
        yaml.dump(result_with_metrics, outfile, default_flow_style=False, sort_keys=False, allow_unicode=True)