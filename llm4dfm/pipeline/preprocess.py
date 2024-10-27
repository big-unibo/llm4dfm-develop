from llm4dfm.pipeline.utils import load_yaml_from_resources

def _process(deps, ignore, substitutions):
    dep = []
    for d in deps:
        d = d.replace(' ', '')
        if d.lower() in ignore:
            dep = []
            break
        dep_to_add = d
        for word, sub_words in substitutions.items():
            if d != word and d.lower() in sub_words:
                dep_to_add = word
                break
        dep.append(dep_to_add)
    return ','.join(dep)


def get_dict_to_check(common, exercise):
    #print(exercise)
    common_values = [list_val for sub_dict in common for list_val in sub_dict.values()]
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
    return {key: [val.lower() for val in value] for my_dict in list_dict for key, value in my_dict.items()}




def preprocess(ex_number, dependencies, measures, fact, demand):

    prep = load_yaml_from_resources('preprocess')

    key = 'demand' if demand else 'supply'

    prep[ex_number] = prep[ex_number][key]
    prep['common'] = prep[key]['common']

    eq_common = prep['common']['equals'] if 'equals' in prep['common'] else []
    eq_ex = prep[ex_number]['equals'] if 'equals' in prep[ex_number] else []
    eq_dicts_to_check = get_dict_to_check(eq_common, eq_ex)

    ignore_common = prep['common']['ignore'] if 'ignore' in prep['common'] else []
    ignore_ex = prep[ex_number]['ignore'] if 'ignore' in prep[ex_number] else []
    ignore = ignore_common + ignore_ex
    ignore_to_check = [ig.lower() for ig in ignore]

    dep_preprocessed = []

    for dep in dependencies:

        frag_dep = {'from': dep['from'].split(','),
                    'to': dep['to'].split(',')}
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
            if 'role' in dep:
                dep_preprocessed.append({'from': dep_from, 'to': dep_to, 'role': dep['role']})
            else:
                dep_preprocessed.append({'from': dep_from, 'to': dep_to})

    meas_preprocessed = []

    for meas in measures:
        meas_preprocessed.append({'name': _process([meas['name']], ignore_to_check, eq_dicts_to_check)})

    measures = meas_preprocessed

    fact = {'name': _process([fact['name']], ignore_to_check, eq_dicts_to_check)}

    return dep_preprocessed, measures, fact
