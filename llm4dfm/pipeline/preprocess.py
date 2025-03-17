from llm4dfm.pipeline.utils import load_yaml_from_resources

def erase(string, chars_to_remove):
    string_to_ret = string
    for char in chars_to_remove:
        string_to_ret = string_to_ret.replace(char, "")
    return string_to_ret

def _process(deps, ignore, substitutions):
    dep = []
    for d in deps:
        d = erase(d, [' ', '-', '_'])
        if d.lower() in ignore:
            dep = []
            break
        dep_to_add = d
        for word, sub_words in substitutions.items():
            if d != word and d.lower() in sub_words:
                dep_to_add = word
                break
        dep.append(erase(dep_to_add, [' ', '-', '_']))
    return ','.join(dep)


def get_dict_to_check(common, exercise):
    common_values = [list_val  for sub_dict in common for list_val in sub_dict.values()]
    common_ex = [list_val for sub_dict in exercise for list_val in sub_dict.values()]

    list_dict = exercise

    for idx_list, common_values_list in enumerate(common_values):
        idx_to_insert = []
        for idx_val, common_val in enumerate(common_values_list):
            insert = True
            for common_ex_list in common_ex:
                if common_val in common_ex_list:
                    insert = False
            if insert:
                idx_to_insert.append(idx_val)
        if len(idx_to_insert) > 0:
            if len(idx_to_insert) == len(common_values_list):
                list_dict.append(common[idx_list])
    return {erase(key, [' ', '-', '_']): [erase(val.lower(), [' ', '-', '_']) for val in value] for my_dict in list_dict for key, value in my_dict.items()}


# Need to search in both from and to
def convert_same_nodes_different_order(node, node_preprocessed):
    if (node != node_preprocessed and {node_prep for node_prep in node_preprocessed.split(',')} ==
            {node_standard for node_standard in node.split(',')} and len(node_preprocessed.split(',')) ==
            len(node.split(','))):
        return node_preprocessed
    else:
        return node


# Convention to get output and ground truth following same order convention in nodes
def preprocess(ex_number, dependencies, measures, fact, demand, nodes_convention_list=None):
    if nodes_convention_list is None:
        nodes_convention_list = list()

    prep = load_yaml_from_resources('preprocess')

    key = 'demand' if demand else 'supply'

    prep[ex_number] = prep[ex_number][key] if ex_number in prep else dict()
    prep['common'] = prep['common'][key]

    eq_common = prep['common']['equals'] if 'equals' in prep['common'] else []
    eq_ex = prep[ex_number]['equals'] if 'equals' in prep[ex_number] else []

    eq_dicts_to_check = get_dict_to_check(eq_common, eq_ex)

    ignore_common = prep['common']['ignore'] if 'ignore' in prep['common'] else []
    ignore_ex = prep[ex_number]['ignore'] if 'ignore' in prep[ex_number] else []
    ignore = ignore_common + ignore_ex
    ignore_to_check = [ig.lower() for ig in ignore]

    dep_preprocessed = []

    for dep in dependencies:
        frag_dep = {'from': dep['from'].split(',') if dep and 'from' in dep and dep['from'] else 'ERROR',
                    'to': dep['to'].split(',') if dep and 'to' in dep and dep['to'] else 'ERROR'}
        if demand:
            # if demand check attributes, so split on '.'
            dep_from = _process([single_part for item in frag_dep['from'] for single_part in item.split('.')],
                                ignore_to_check, eq_dicts_to_check)
            dep_to = _process([single_part for item in frag_dep['to'] for single_part in item.split('.')],
                           ignore_to_check, eq_dicts_to_check)
        else:
            # if supply check the table name too, so not split on '.'
            dep_from = _process([item for item in frag_dep['from']], ignore_to_check, eq_dicts_to_check)
            dep_to = _process([item for item in frag_dep['to']], ignore_to_check, eq_dicts_to_check)
        if dep_from and dep_to:
            from_before = dep_from
            to_before = dep_to
            # Convert all the nodes which must be considered the same to the first occurrence of the node
            for prep_dict in nodes_convention_list + dep_preprocessed:
                dep_from = convert_same_nodes_different_order(dep_from, prep_dict['from'])
                if from_before == dep_from:
                    dep_from = convert_same_nodes_different_order(dep_from, prep_dict['to'])

                dep_to = convert_same_nodes_different_order(dep_to, prep_dict['from'])
                if to_before == dep_to:
                    dep_to = convert_same_nodes_different_order(dep_to, prep_dict['to'])

                if from_before != dep_from and to_before != dep_to:
                    break
            if dep and 'role' in dep:
                dep_preprocessed.append({'from': dep_from, 'to': dep_to, 'role': dep['role']})
            else:
                dep_preprocessed.append({'from': dep_from, 'to': dep_to})

    meas_preprocessed = []

    for meas in measures:
        meas_preprocessed.append({'name': f'{_process([meas['name']], ignore_to_check, eq_dicts_to_check)}'})

    measures = [meas for meas in meas_preprocessed if meas['name']]

    fact = {'name': f'{_process([fact['name']], ignore_to_check, eq_dicts_to_check)}'}

    return dep_preprocessed, measures, fact
