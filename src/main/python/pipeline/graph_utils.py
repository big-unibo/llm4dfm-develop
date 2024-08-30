

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


# Filter dict of ground truth if created
def is_a_valid_dependency(dependency_dict):
    if 'refinements' in dependency_dict:
        return dependency_dict['refinements'] != 'created'
    return True

# Calculates metrics from ground_truth set and generated set
def get_metrics_nodes(dep_gt, dep_generated, measure_gt, measure_generated, fact_gt, fact_generated):
    print(f'dep - gt: {dep_gt} {len(dep_gt)}\n')
    print(f'dep - out: {dep_generated} {len(dep_generated)}\n')

    tp_dep = dep_gt & dep_generated
    fn_dep = dep_gt - tp_dep
    fp_dep = dep_generated - tp_dep

    print(f'tp {tp_dep} {len(tp_dep)}')
    print(f'fn {fn_dep} {len(fn_dep)}')
    print(f'fp {fp_dep} {len(fp_dep)}\n')

    print(f'meas - gt: {measure_gt} {len(measure_gt)}\n')
    print(f'meas - out: {measure_generated} {len(measure_generated)}\n')

    tp_meas = measure_gt & measure_generated
    fn_meas = measure_gt - tp_meas
    fp_meas = measure_generated - tp_meas

    print(f'tp {tp_meas} {len(tp_meas)}')
    print(f'fn {fn_meas} {len(fn_meas)}')
    print(f'fp {fp_meas} {len(fp_meas)}\n')

    tp_fact = 1 if fact_gt == fact_generated else 0

    print(f'fact gt {fact_gt} generated {fact_generated} -> {tp_fact}')

    # JS consider valid the PURCHASE given that it's correctly fact, but not found in deps
    # JS doesn't consider overlapping in measures and dependencies

    tp_count = len(tp_dep) + len(tp_meas) + tp_fact
    fn_count = len(fn_dep) + len(fn_meas) + (1 - tp_fact)
    fp_count = len(fp_dep) + len(fp_meas) + (1 - tp_fact)

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

    return precision, recall, f1, tp_count, fn_count, fp_count

# Calculates metrics from ground_truth set and generated set
def get_metrics_edges(ground_truth, generated):
    tp = ground_truth & generated
    fn = ground_truth - tp
    fp = generated - tp

    tp_count = len(tp)
    fn_count = len(fn)
    fp_count = len(fp)

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

    return precision, recall, f1, tp_count, fn_count, fp_count
