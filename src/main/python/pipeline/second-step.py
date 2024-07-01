from pathlib import Path
from utils import load_yaml, load_ground_truth_exercise, load_output_exercise
from ssutils import clean_gt_dependencies, remove_explicit_tables_to_output, is_a_valid_role_dependency
import networkx as nx
import matplotlib.pyplot as plt

input_config = load_yaml(f'{Path().absolute()}/pipeline/second-step-config.yml')

ex_config = input_config['exercise']
model_config = input_config['model']
output_config = input_config['output']

ex_output = load_output_exercise(ex_config['name'], ex_config['v'], ex_config['prompt_v'],
                                 model_config['name'], model_config['v'],
                                 output_config['latest'], output_config['timestamp'], ex_config['full_name'])

ground_truth = load_ground_truth_exercise(input_config['exercise']['name'])

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

# Create a directed graph
G = nx.DiGraph()

tp_color, fn_color, fp_color = 'green', 'grey', 'red'

# Add nodes and edges from the dictionaries
for d in tp_list:
    dep_dict = dict()
    for key, value in d.items():
        dep_dict[key] = value
        G.add_node(value)
    G.add_edge(dep_dict['from'], dep_dict['to'], color=tp_color)

for d in fn_list:
    dep_dict = dict()
    for key, value in d.items():
        dep_dict[key] = value
        G.add_node(value)
    G.add_edge(dep_dict['from'], dep_dict['to'], color=fn_color)

for d in fp_list:
    dep_dict = dict()
    for key, value in d.items():
        dep_dict[key] = value
        G.add_node(value)
    G.add_edge(dep_dict['from'], dep_dict['to'], color=fp_color)


# Draw the graph
pos = nx.spring_layout(G, seed=42)  # positions for all nodes
plt.figure(figsize=(10, 8))

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]

# Draw nodes and edges
nx.draw(G, pos, edge_color=colors, with_labels=True, node_color='lightblue', node_size=1000,
        font_size=8, font_weight='bold', arrows=True)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# TODO not working
plt.text(0.5, 1.05, f'TP: {tp_count}\nFN: {fn_count}\nFP: {fp_count}', horizontalalignment='center',
         fontsize=14, transform=plt.gca().transAxes)

# Display the graph
plt.title("Graph Visualization")
plt.show()
