import matplotlib.pyplot as plt
import numpy as np
from utils import auto_outputs

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
    tp_dep = {dep.lower() for dep in dep_gt & dep_generated}
    fn_dep = {dep.lower() for dep in dep_gt - tp_dep}
    fp_dep = {dep.lower() for dep in dep_generated - tp_dep}

    tp_meas = {mes.lower() for mes in measure_gt & measure_generated}
    fn_meas = {mes.lower() for mes in measure_gt - tp_meas}
    fp_meas = {mes.lower() for mes in measure_generated - tp_meas}

    fact_gt_use = fact_gt.lower()
    fact_generated_use = fact_generated.lower()

    tp = tp_dep.union(tp_meas)
    fn = fn_dep.union(fn_meas)
    fp = fp_dep.union(fp_meas)

    if fact_gt_use == fact_generated_use:
        tp.add(fact_gt_use)
    else:
        fn.add(fact_gt_use)
        fp.add(fact_generated_use)

    # Remove intersection
    fn -= tp
    fp -= tp

    tp_count = len(tp)
    fn_count = len(fn)
    fp_count = len(fp)

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

def plot_csv_metrics(data, file_name):
    # Function to calculate average
    def calculate_average(values):
        values = [float(v) for v in values]
        return sum(values) / len(values)

    # Prepare data for plotting
    exercises = list(data.keys())
    edges_precision_avg = [calculate_average(data[ex]['edges_precision']) for ex in exercises]
    edges_recall_avg = [calculate_average(data[ex]['edges_recall']) for ex in exercises]
    nodes_precision_avg = [calculate_average(data[ex]['nodes_precision']) for ex in exercises]
    nodes_recall_avg = [calculate_average(data[ex]['nodes_recall']) for ex in exercises]
    edges_f1 = [list(map(float, data[ex]['edges_f1'])) for ex in exercises]
    nodes_f1 = [list(map(float, data[ex]['nodes_f1'])) for ex in exercises]

    ex_indexes = [ex.split('-')[-1] for ex in exercises]

    bar_width = 0.35
    index = np.arange(len(exercises))

    prec_color = 'black'
    rec_color = 'red'
    f1_edges_color = 'red'
    f1_nodes_color = 'black'

    ax_limits = [0,1]

    # Plot 1: Avg Precision and Recall for Edges
    fig1, ax1 = plt.subplots()
    ax1.bar(index - bar_width / 2, edges_precision_avg, bar_width, label='Precision', color=prec_color)
    ax1.bar(index + bar_width / 2, edges_recall_avg, bar_width, label='Recall', color=rec_color)
    ax1.set_xlabel('Exercise')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Average Precision and Recall for Edges')
    ax1.set_xticks(index)
    ax1.set_xticklabels(ex_indexes)
    ax1.set_ylim(ax_limits)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{file_name}/graph-precision_recall_edges.pdf", format='pdf')

    # Plot 2: Avg Precision and Recall for Nodes
    fig2, ax2 = plt.subplots()
    ax2.bar(index - bar_width / 2, nodes_precision_avg, bar_width, label='Precision', color=prec_color)
    ax2.bar(index + bar_width / 2, nodes_recall_avg, bar_width, label='Recall', color=rec_color)
    ax2.set_xlabel('Exercise')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Average Precision and Recall for Nodes')
    ax2.set_xticks(index)
    ax2.set_xticklabels(ex_indexes)
    ax2.set_ylim(ax_limits)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{file_name}/graph-precision_recall_nodes.pdf", format='pdf')

    # Calculate average F1 scores for the line chart
    edges_f1_avg = [calculate_average(data[ex]['edges_f1']) for ex in exercises]
    nodes_f1_avg = [calculate_average(data[ex]['nodes_f1']) for ex in exercises]

    # Plot 3: Line chart of Avg F1 Scores for Edges and Nodes
    fig3, ax3 = plt.subplots()
    ax3.plot(ex_indexes, edges_f1_avg, color=f1_edges_color, label='Avg F1 of Edges')
    ax3.plot(ex_indexes, nodes_f1_avg, color=f1_nodes_color, label='Avg F1 of Nodes')
    ax3.set_xlabel('Exercise')
    ax3.set_ylabel('Average F1 Score')
    ax3.set_title('Average F1 Score for Edges and Nodes')
    ax3.set_ylim(ax_limits)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(f"{file_name}/graph-f1_scores_edges_nodes.pdf", format='pdf')

    # Plot 4: Boxplot of F1 Measure for Edges
    fig4, ax4 = plt.subplots()
    ax4.boxplot(edges_f1, labels=ex_indexes)
    ax4.set_xlabel('Exercise')
    ax4.set_ylabel('F1 Score')
    ax4.set_ylim(ax_limits)
    ax4.set_title('Boxplot of F1 Measure for Edges')
    fig4.tight_layout()
    fig4.savefig(f"{file_name}/graph-boxplot_f1_edges.pdf", format='pdf')

    # Plot 5: Boxplot of F1 Measure for Nodes
    fig5, ax5 = plt.subplots()
    ax5.boxplot(nodes_f1, labels=ex_indexes)
    ax5.set_xlabel('Exercise')
    ax5.set_ylabel('F1 Score')
    ax5.set_ylim(ax_limits)
    ax5.set_title('Boxplot of F1 Measure for Nodes')
    fig5.tight_layout()
    fig5.savefig(f"{file_name}/graph-boxplot_f1_nodes.pdf", format='pdf')

    # Set up the figure and subplots
    # fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    # # Plot 1: Avg Precision and Recall for Edges
    # axs[0, 0].bar(index - bar_width / 2, edges_precision_avg, bar_width, label='Precision', color=prec_color)
    # axs[0, 0].bar(index + bar_width / 2, edges_recall_avg, bar_width, label='Recall', color=rec_color)
    # axs[0, 0].set_xlabel('Exercise')
    # axs[0, 0].set_ylabel('Average Score')
    # axs[0, 0].set_title('Average Precision and Recall for Edges')
    # axs[0, 0].set_xticks(index)
    # axs[0, 0].set_xticklabels(ex_indexes)
    # axs[0, 0].set_ylim(ax_limits)
    # axs[0, 0].legend()
    #
    # # Plot 2: Avg Precision and Recall for Nodes
    # axs[0, 1].bar(index - bar_width / 2, nodes_precision_avg, bar_width, label='Precision', color=prec_color)
    # axs[0, 1].bar(index + bar_width / 2, nodes_recall_avg, bar_width, label='Recall', color=rec_color)
    # axs[0, 1].set_xlabel('Exercise')
    # axs[0, 1].set_ylabel('Average Score')
    # axs[0, 1].set_title('Average Precision and Recall for Nodes')
    # axs[0, 1].set_xticks(index)
    # axs[0, 1].set_xticklabels(ex_indexes)
    # axs[0, 1].set_ylim(ax_limits)
    # axs[0, 1].legend()
    #
    # # Calculate average F1 scores for the line chart
    # edges_f1_avg = [calculate_average(data[ex]['edges_f1']) for ex in exercises]
    # nodes_f1_avg = [calculate_average(data[ex]['nodes_f1']) for ex in exercises]
    #
    # # Plot 3: Line chart of Avg F1 Scores for Edges and Nodes
    # axs[0, 2].plot(ex_indexes, edges_f1_avg, color=f1_edges_color, label='Avg F1 of Edges')
    # axs[0, 2].plot(ex_indexes, nodes_f1_avg, color=f1_nodes_color, label='Avg F1 of Nodes')
    # axs[0, 2].set_xlabel('Exercise')
    # axs[0, 2].set_ylabel('Average F1 Score')
    # axs[0, 2].set_title('Average F1 Score for Edges and Nodes')
    # axs[0, 2].set_ylim(ax_limits)
    # axs[0, 2].legend()
    #
    # # Plot 4: Boxplot of F1 Measure for Edges
    # axs[1, 0].boxplot(edges_f1, labels=ex_indexes)
    # axs[1, 0].set_xlabel('Exercise')
    # axs[1, 0].set_ylabel('F1 Score')
    # axs[1, 0].set_ylim(ax_limits)
    # axs[1, 0].set_title('Boxplot of F1 Measure for Edges')
    #
    # # Plot 5: Boxplot of F1 Measure for Nodes
    # axs[1, 1].boxplot(nodes_f1, labels=ex_indexes)
    # axs[1, 1].set_xlabel('Exercise')
    # axs[1, 1].set_ylabel('F1 Score')
    # axs[1, 1].set_ylim(ax_limits)
    # axs[1, 1].set_title('Boxplot of F1 Measure for Nodes')
    #
    # # Hide the empty subplot (bottom right)
    # axs[1, 2].axis('off')
    # # Save the figure as a PDF
    # plt.savefig(f"{file_name}/my-graph.pdf", format='pdf')
    #
    # # Adjust layout
    # plt.tight_layout()
    # plt.show()
