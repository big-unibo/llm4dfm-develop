from utils import load_yaml_from_resources

def _process(coll, ignore, dicts):
    dep = []
    for d in coll:
        d = d.replace(' ', '')
        if d.lower() in ignore:
            print(f'{d} in ignore, breaking')
            dep = []
            break
        dep_to_add = d
        for eq_dict in dicts:
            if d.lower() in eq_dict.values():
                for k, v in eq_dict.items():
                    if d.lower() in v:
                        dep_to_add = k
                        break
                print(f'Dep_from -> {d.lower()} changed to {dep_to_add}')
        dep.append(dep_to_add)

    return dep


def preprocess(ex_number, dependencies, measures, fact, demand):
    prep = load_yaml_from_resources('preprocess')

    key = 'demand' if demand else 'supply'

    prep[ex_number] = prep[ex_number][key]
    prep['common'] = prep[key]['common']

    eq_common = prep['common']['equals'] if 'equals' in prep['common'] else []
    eq_ex = prep[ex_number]['equals'] if 'equals' in prep[ex_number] else []
    eq = eq_common + eq_ex
    eq_dicts_to_check = [{key: [val.lower() for val in value] for key, value in my_dict.items()} for my_dict in eq]

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
            dep_from = '.'.join(_process([single_part for item in frag_dep['from'] for single_part in item.split('.')],
                                ignore_to_check, eq_dicts_to_check))
            dep_to = '.'.join(_process([single_part for item in frag_dep['to'] for single_part in item.split('.')],
                           ignore_to_check, eq_dicts_to_check))
        else:
            # if supply check the table name too, so not split on '.'
            dep_from = ','.join(_process([item for item in frag_dep['from']], ignore_to_check, eq_dicts_to_check))
            dep_to = ','.join(_process([item for item in frag_dep['to']], ignore_to_check, eq_dicts_to_check))
        if dep_from and dep_to:
            if 'role' in dep:
                dep_preprocessed.append({'from': dep_from, 'to': dep_to, 'role': dep['role']})
            else:
                dep_preprocessed.append({'from': dep_from, 'to': dep_to})

    return dep_preprocessed, measures, fact
