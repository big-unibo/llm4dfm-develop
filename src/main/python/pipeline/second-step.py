from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.lines as mlines

from ssutils import (preprocess_dependencies_attributes, load_edges, load_nodes, store_image, short_names_from_tables,
                     get_metrics, get_tp_fn_fp_edges_to_list, update_output_with_metrics)
from utils import load_yaml, load_ground_truth_exercise, load_output_exercise_and_name

# Load config
input_config = load_yaml(f'{Path().absolute()}/pipeline/second-step-config.yml')
ex_config = input_config['exercise']
model_config = input_config['model']

# Load exercise
ex_output, ex_name = load_output_exercise_and_name(ex_config['name'], ex_config['v'], ex_config['prompt_v'],
                                 model_config['name'], model_config['v'],
                                 ex_config['latest'], ex_config['timestamp'], ex_config['full_name'])
ground_truth = load_ground_truth_exercise(ex_config['name'], ex_config['full_name'])

# Extract dependencies
dep_output = ex_output['output']['dependencies'] if ex_output['output'] is dict else ex_output['output'][0]['dependencies']
dep_gt = ground_truth['dependencies']

# Load edges
edges_set_gt = load_edges(dep_gt)
edges_set_output = load_edges(dep_output)

# Load nodes
nodes_set_gt = load_nodes(edges_set_gt)
nodes_set_output = load_nodes(edges_set_output)

# Calculate metrics for edges and ground truth
precision_edges, recall_edges, f1_edges = get_metrics(edges_set_gt, edges_set_output)
precision_nodes, recall_nodes, f1_nodes = get_metrics(nodes_set_gt, nodes_set_output)

metrics = {
    'edges': {
        'precision': round(precision_edges * 100, 2),
        'recall': round(recall_edges * 100, 2),
        'f1': round(f1_edges * 100, 2),
    },
    'nodes': {
        'precision': round(precision_nodes * 100, 2),
        'recall': round(recall_nodes * 100, 2),
        'f1': round(f1_nodes * 100, 2),
    }
}

# TODO add fact visualization
# fact = ground_truth['fact']['key'] if 'fact' in ground_truth else ''

# Visualization

G = nx.DiGraph()

tp_color, fn_color, fp_color = 'green', 'grey', 'red'
dep_loop_color = 'yellow'

# Extract tables name to obtain short names
short_names = short_names_from_tables(edges_set_gt, edges_set_output)

# List of edges classified to color correctly
tp_edges_list, fn_edges_list, fp_edges_list = get_tp_fn_fp_edges_to_list(edges_set_gt, edges_set_output)

inserted_nodes = []

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
    for component in metrics:
        legend_items.append(mlines.Line2D([], [], color='black', linestyle='-', linewidth=2))
        legend_items.extend([plt.Line2D([0], [0], color='w', label=f'{component.capitalize()}:')])
        for sub_metrics in metrics[component]:
            legend_items.extend([plt.Line2D([0], [0], color='w', label=f'{sub_metrics}: {metrics[component][sub_metrics]}%')])
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

if 'metrics' not in ex_output:
    ex_output['metrics'] = metrics
    update_output_with_metrics(ex_name, ex_output)
