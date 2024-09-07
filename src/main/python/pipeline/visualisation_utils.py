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
def store_image(plt, name, dir_name, img_format):
    plt.savefig(f'{outputs}{dir_name}/{Path(name).stem}.{img_format}', format=img_format)
