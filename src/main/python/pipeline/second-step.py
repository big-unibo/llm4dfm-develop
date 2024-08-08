import collections
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.lines as mlines

from ssutils import (preprocess_dependencies_attributes, is_a_valid_role_dependency, is_a_valid_dependency,
                     store_image, short_names_from_tables, get_clean_table_attribute)
from utils import load_yaml, load_ground_truth_exercise, load_output_exercise_and_name

input_config = load_yaml(f'{Path().absolute()}/pipeline/second-step-config.yml')

ex_config = input_config['exercise']
model_config = input_config['model']

ex_output, ex_name = load_output_exercise_and_name(ex_config['name'], ex_config['v'], ex_config['prompt_v'],
                                 model_config['name'], model_config['v'],
                                 ex_config['latest'], ex_config['timestamp'], ex_config['full_name'])

ground_truth = load_ground_truth_exercise(ex_config['name'], ex_config['full_name'])

dep_output = ex_output['output']['dependencies'] if ex_output['output'] is dict else ex_output['output'][0]['dependencies']

dep_gt = ground_truth['dependencies']

tables = []

edges_set_gt = set(
    frozenset((key, get_clean_table_attribute(value))
              for key, value in d.items() if is_a_valid_role_dependency(key))
    for d in dep_gt if is_a_valid_dependency(d))
edges_set_output = set(
    frozenset((key, get_clean_table_attribute(value))
              for key, value in d.items())
    for d in dep_output)

nodes_set_gt = set(
    get_clean_table_attribute(entry[1])
    for fr_set in edges_set_gt
    for entry in fr_set
)

nodes_set_output = set(
    get_clean_table_attribute(entry[1])
    for fr_set in edges_set_output
    for entry in fr_set
)

tp_edges = edges_set_gt & edges_set_output
fn_edges = edges_set_gt - tp_edges
fp_edges = edges_set_output - tp_edges

tp_nodes = nodes_set_gt & nodes_set_output
fn_nodes = nodes_set_gt - tp_nodes
fp_nodes = nodes_set_output - tp_nodes

tp_edges_count = len(tp_edges)
fn_edges_count = len(fn_edges)
fp_edges_count = len(fp_edges)

tp_nodes_count = len(tp_nodes)
fn_nodes_count = len(fn_nodes)
fp_nodes_count = len(fp_nodes)

precision_edges = tp_edges_count / (tp_edges_count + fp_edges_count)
recall_edges = tp_edges_count / (tp_edges_count + fn_edges_count)
f1_edges = 2 * ((precision_edges * recall_edges) / (precision_edges + recall_edges)) if precision_edges + recall_edges != 0 else 0

precision_nodes = tp_nodes_count / (tp_nodes_count + fp_nodes_count)
recall_nodes = tp_nodes_count / (tp_nodes_count + fn_nodes_count)
f1_nodes = 2 * ((precision_nodes * recall_nodes) / (precision_nodes + recall_nodes)) if precision_nodes + recall_nodes != 0 else 0

metrics = {
    'precision_edges': round(precision_edges * 100, 2),
    'recall_edges': round(recall_edges * 100, 2),
    'f1_edges': round(f1_edges * 100, 2),
    'precision_nodes': round(precision_nodes * 100, 2),
    'recall_nodes': round(recall_nodes * 100, 2),
    'f1_nodes': round(f1_nodes * 100, 2),
}

# TODO add fact visualization
# fact = ground_truth['fact']['key'] if 'fact' in ground_truth else ''

# Visualization

G = nx.DiGraph()

tp_color, fn_color, fp_color = 'green', 'grey', 'red'
dep_loop_color = 'yellow'

inserted_nodes = []

# Extract tables name to obtain short names
short_names = short_names_from_tables(edges_set_gt, edges_set_output)

# List of edges classified to color correctly
tp_edges_list = [collections.OrderedDict(sorted(fs)) for fs in tp_edges]
tp_edges_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))
fn_edges_list = [collections.OrderedDict(sorted(fs)) for fs in fn_edges]
fn_edges_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))
fp_edges_list = [collections.OrderedDict(sorted(fs)) for fs in fp_edges]
fp_edges_list.sort(key=lambda dependency: (dependency['from'], dependency['to']))

# Add nodes and edges from the dictionaries
for dep_list in [tp_edges_list, fn_edges_list, fp_edges_list]:
    for dep in dep_list:
        dep_dict = dict()
        color = tp_color if dep in tp_edges_list else fn_color if dep in fn_edges_list else fp_color if dep in fp_edges_list else \
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
                if dep in fp_edges_list:
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

if input_config['visualization']['table_names']:
    legend_items = [plt.Line2D([0], [0], color='w', label=f'{full_name}: {short_name}')
                    for full_name, short_name in short_names.items()]
    legend_items.append(mlines.Line2D([], [], color='black', linestyle='-', linewidth=2))
    legend_items.extend([plt.Line2D([0], [0], color='w', label=f'{label}: {str(value)}%')
     for label, value in metrics.items()])
    plt.legend(handles=legend_items, title="Tables convention", fontsize='small', title_fontsize='medium',
               labelspacing=0.3, handletextpad=0.4, loc='upper left')

# Display the grap
plt.title("Graph Visualization")

if input_config['visualization']['image']['generate']:
    store_image(plt, ex_name, input_config['visualization']['image']['format'])

if input_config['visualization']['show_graph']:
    plt.show()
else:
    # Close the plot
    plt.close()
