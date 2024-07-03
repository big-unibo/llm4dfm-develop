from pathlib import Path
from utils import load_yaml, load_ground_truth_exercise, load_output_exercise_and_name
from ssutils import clean_gt_dependencies, remove_explicit_tables_to_output, is_a_valid_role_dependency, store_image
import networkx as nx
import matplotlib.pyplot as plt

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

tp_list = [dict(fs) for fs in tp]
fn_list = [dict(fs) for fs in fn]
fp_list = [dict(fs) for fs in fp]

# print(f'TP: {tp_list}\n\nFN: {fn_list}\n\nFP: {fp_list}')

tp_count = len(tp)
fn_count = len(fn)
fp_count = len(fp)

# print(f"TP: {tp_count}\nFN: {fn_count}\nFP: {fp_count}")

# Visualization

G = nx.DiGraph()

tp_color, fn_color, fp_color = 'green', 'grey', 'red'

inserted_nodes = []

# Add nodes and edges from the dictionaries
for dep_list in [tp_list, fn_list, fp_list]:
    for dep in dep_list:
        dep_dict = dict()
        color = tp_color if dep in tp_list else fn_color if dep in fn_list else fp_color if dep in fp_list else \
            'black'
        for key, value in dep.items():
            dep_dict[key] = value
            if value not in inserted_nodes:
                G.add_node(value.replace(',', '\n'), color=color)
                inserted_nodes.append(value)
        G.add_edge(dep_dict['from'].replace(',', '\n'), dep_dict['to'].replace(',', '\n'), color=color)

# Draw the graph
k = input_config['visualization']['k']
arrowsize = input_config['visualization']['arrowsize']
pos = nx.spring_layout(G, seed=42, k=k)  # positions for all nodes
plt.figure(figsize=(10, 8))

edge_colors = [G[u][v]['color'] for u, v in G.edges()] if input_config['visualization']['edge_color'] else None
node_colors = [G.nodes[n]['color'] for n in G.nodes()] if input_config['visualization']['node_color'] else None

# Draw nodes and edges
nx.draw(G, pos, edge_color=edge_colors, node_color=node_colors,
        with_labels=True, node_size=1000,
        font_size=10, font_weight='bold', arrows=True, arrowsize=arrowsize)

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
