from pathlib import Path
from utils import load_yaml, load_ground_truth_exercise, load_output_exercise_and_name
from ssutils import clean_gt_dependencies, remove_explicit_tables_to_output, is_a_valid_role_dependency, store_image
import networkx as nx
import matplotlib.pyplot as plt
import collections
import pygraphviz

input_config = load_yaml(f'{Path().absolute()}/pipeline/second-step-config.yml')

ex_config = input_config['exercise']
model_config = input_config['model']

ex_output, ex_name = load_output_exercise_and_name(ex_config['name'], ex_config['v'], ex_config['prompt_v'],
                                 model_config['name'], model_config['v'],
                                 ex_config['latest'], ex_config['timestamp'], ex_config['full_name'])

ground_truth = load_ground_truth_exercise(ex_config['name'], ex_config['full_name'])

dep_output = ex_output['output'][0]['dependencies']

# Given a different format ground-truth and model-output, uniform it
dep_gt_dirty = ground_truth[0]['dependencies']
dep_gt = clean_gt_dependencies(dep_gt_dirty)

set_gt = set(
    frozenset((key, value) for key, value in d.items() if is_a_valid_role_dependency(key))
    for d in dep_gt)
set_output = set(
    frozenset((key, remove_explicit_tables_to_output(value))
              for key, value in d.items())
    for d in dep_output)

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

fact = ground_truth[1]['fact'][1]['key']

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
            # A node must be added only one time
            if value not in inserted_nodes:
                G.add_node(value.replace(',', '\n'), color=color)
                inserted_nodes.append(value)
            # If already present, in case of dependency in false positive (red one), if already considered
            # in false negative (grey), must be converted to true positive since it's been detected as
            # a dependency (green)
            else:
                if dep in fp_list:
                    G.nodes[value.replace(',', '\n')]['color'] = tp_color
        # If it's not auto dependency can be added
        if not input_config['visualization']['dag_visualization'] or dep_dict['from'] != dep_dict['to']:
            G.add_edge(dep_dict['from'].replace(',', '\n'), dep_dict['to'].replace(',', '\n'), color=color)
        # Otherwise it's not added and color is changed to yellow, given that graph visualization is based on DAG
        else:
            G.nodes[dep_dict['from'].replace(',', '\n')]['color'] = dep_loop_color

arrow_size = input_config['visualization']['arrow_size']

if input_config['visualization']['dag_visualization']:
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

# Display the graph
plt.title("Graph Visualization")

if input_config['visualization']['image']['generate']:
    store_image(plt, ex_name, input_config['visualization']['image']['format'])

if input_config['visualization']['show_graph']:
    plt.show()
else:
    # Close the plot
    plt.close()
