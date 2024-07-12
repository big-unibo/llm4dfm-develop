import collections
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from ssutils import preprocess_dependencies_attributes, is_a_valid_role_dependency, store_image
from utils import load_yaml, load_ground_truth_exercise, load_output_exercise_and_name

input_config = load_yaml(f'{Path().absolute()}/pipeline/second-step-config.yml')

ex_config = input_config['exercise']
model_config = input_config['model']

ex_output, ex_name = load_output_exercise_and_name(ex_config['name'], ex_config['v'], ex_config['prompt_v'],
                                 model_config['name'], model_config['v'],
                                 ex_config['latest'], ex_config['timestamp'], ex_config['full_name'])

ground_truth = load_ground_truth_exercise(ex_config['name'], ex_config['full_name'])

dep_output = ex_output['output']['dependencies']

dep_gt = ground_truth['dependencies']

tables = []

set_gt = set(
    frozenset((key, value)
              for key, value in d.items() if is_a_valid_role_dependency(key))
    for d in dep_gt)
set_output = set(
    frozenset((key, value)
              for key, value in d.items())
    for d in dep_output)

all_tables_gt = set(
    val.split('.')[0].replace(' ', '')
    for subset in set_gt
    for _, value in subset
    for val in value.split(',')
    if '.' in val
)

all_tables_out = set(
    val.split('.')[0].replace(' ', '')
    for subset in set_output
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

tp = set_gt & set_output
fn = set_gt - tp
fp = set_output - tp

tp_list = [collections.OrderedDict(sorted(fs)) for fs in tp]
tp_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))
fn_list = [collections.OrderedDict(sorted(fs)) for fs in fn]
fn_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))
fp_list = [collections.OrderedDict(sorted(fs)) for fs in fp]
fp_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))

# print(f'TP: {tp_list}\n\nFN: {fn_list}\n\nFP: {fp_list}')

tp_count = len(tp)
fn_count = len(fn)
fp_count = len(fp)

# print(f"TP: {tp_count}\nFN: {fn_count}\nFP: {fp_count}")

fact = ground_truth['fact']['key']

# Visualization

G = nx.DiGraph()

tp_color, fn_color, fp_color = 'green', 'grey', 'red'
dep_loop_color = 'yellow'

inserted_nodes = []


# Add nodes and edges from the dictionaries
for dep_list in [tp_list, fn_list, fp_list]:
    for dep in dep_list:
        dep_dict = dict()
        color = tp_color if dep in tp_list else fn_color if dep in fn_list else fp_color if dep in fp_list else \
            'black'
        for key, value in dep.items():
            dep_dict[key] = value

            value_preprocessed = preprocess_dependencies_attributes(value, input_config['visualization']['table_names'], short_names)

            if value_preprocessed not in inserted_nodes:
                G.add_node(value_preprocessed, color=color)
                inserted_nodes.append(value_preprocessed)
            # If already present, in case of dependency in false positive (red one), if already considered
            # in false negative (grey), must be converted to true positive since it's been detected as
            # a dependency (green)
            else:
                if dep in fp_list:
                    G.nodes[value_preprocessed]['color'] = tp_color
        from_preprocessed = preprocess_dependencies_attributes(dep_dict['from'], input_config['visualization']['table_names'], short_names)
        to_preprocessed = preprocess_dependencies_attributes(dep_dict['to'], input_config['visualization']['table_names'], short_names)
        # If it's not auto dependency can be added
        if not input_config['visualization']['dag_graph'] or from_preprocessed != to_preprocessed:
            G.add_edge(from_preprocessed, to_preprocessed, color=color)
        # Otherwise it's not added and color is changed to yellow, given that graph visualization is based on DAG
        else:
            G.nodes[from_preprocessed]['color'] = dep_loop_color

arrow_size = input_config['visualization']['arrow_size']

if input_config['visualization']['dag_graph']:
    if not nx.is_directed_acyclic_graph(G):
        raise Exception('Graph is not DAG, change visualization')
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
else:
    pos = nx.shell_layout(G)

plt.figure(figsize=(12, 8))

edge_colors = [G[u][v]['color'] for u, v in G.edges()] if input_config['visualization']['edge_color'] else None
node_colors = [G.nodes[n]['color'] for n in G.nodes()] if input_config['visualization']['node_color'] else None

# Draw nodes and edges
nx.draw(G, pos, edge_color=edge_colors, node_color=node_colors,
        with_labels=True, node_size=input_config['visualization']['node_size'],
        font_size=input_config['visualization']['font_size'], font_weight='bold', arrows=True, arrowsize=arrow_size)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

legend_items = [plt.Line2D([0], [0], color='w', label=f'{full_name}: {short_name}')
                for full_name, short_name in short_names.items()]

plt.legend(handles=legend_items, title="Tables convention", fontsize='small', title_fontsize='medium',
           labelspacing=0.3, handletextpad=0.4, loc='upper left')

# Display the graph
plt.title("Graph Visualization")

if input_config['visualization']['image']['generate']:
    store_image(plt, ex_name, input_config['visualization']['image']['format'])

if input_config['visualization']['show_graph']:
    plt.show()
else:
    # Close the plot
    plt.close()
