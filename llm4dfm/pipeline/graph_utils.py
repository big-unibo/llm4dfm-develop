import matplotlib.pyplot as plt
import numpy as np


def _calculate_average(values):
    values_converted = []
    for i, value in enumerate(values):
        try:
            values_converted.append(float(value))
        except:
            print(f'Error while converting {i}-th value {value} to float, skipped')
    return sum(values_converted) / len(values_converted)

def _plot_avg(idx, width, val1, val2, col1, col2, lab1, lab2, x_label, y_label, title, x_ticks_labels, y_lim, save_path):
    fig, ax = plt.subplots()
    ax.bar(idx - width / 2, val1, width, label=lab1, color=col1)
    ax.bar(idx + width / 2, val2, width, label=lab2, color=col2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(idx)
    ax.set_xticklabels(x_ticks_labels, ha='right', fontsize='small')
    ax.set_ylim(y_lim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format='pdf')

def _plot_line_chart(idx, val1, val2, col1, col2, lab1, lab2, x_label, y_label, title, y_lim, save_path):
    fig, ax = plt.subplots()
    ax.plot(idx, val1, color=col1, label=lab1)
    ax.plot(idx, val2, color=col2, label=lab2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(range(len(idx)))  # Ensure the ticks align with the number of labels
    ax.set_xticklabels(idx, ha='right', fontsize='small')  # Rotate labels for clarity
    ax.set_ylim(y_lim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format='pdf')

def _plot_boxplot(val, idx, x_label, y_label, title, y_lim, save_path):
    fig, ax = plt.subplots()
    ax.boxplot(val)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(range(1, len(idx) + 1))  # Ensure the ticks align with the number of labels
    ax.set_xticklabels(idx, ha='right', fontsize='small')  # Rotate labels for clarity
    ax.set_ylim(y_lim)
    fig.tight_layout()
    fig.savefig(save_path, format='pdf')

def plot_time_f1(data, file_name, label):

    f1_edges_color = 'red'
    f1_nodes_color = 'black'

    # Compute average F1-score and elapsed time for each exercise

    exercise_stats = data.groupby("ex_name").agg({
        "edges_f1": "mean",
        "nodes_f1": "mean",
        "time": "mean"
    }).reset_index()

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))

    # Scatter plot for edges
    plt.scatter(exercise_stats["time"], exercise_stats["edges_f1"], color=f1_edges_color, label="Edges", alpha=0.7)

    # Scatter plot for nodes
    plt.scatter(exercise_stats["time"], exercise_stats["nodes_f1"], color=f1_nodes_color, label="Nodes", alpha=0.7)

    plt.plot([exercise_stats["time"], exercise_stats["time"]],  # x stays the same
                 [exercise_stats["nodes_f1"], exercise_stats["edges_f1"]],  # y goes from nodes_f1 to edges_f1
                 color='black', linewidth=1, alpha=0.5)

    offset = 0.05

    # Annotate each point with exercise name
    for i, row in exercise_stats.iterrows():
        plt.text(row["time"] - offset, row["nodes_f1"] + offset, row["ex_name"][-1],
                 fontsize=12, ha="right", va="top", color=f1_nodes_color)

    # Labels and title
    plt.xlabel("Average Elapsed Time (s)")
    plt.ylabel("Average F1-score")
    plt.ylim(0, 1.1)
    plt.title(f"F1-score vs. Elapsed Time [{data.at[0, 'config_label']}]")
    plt.xlim(left=0)
    plt.legend()  # Show legend
    plt.grid(True)

    plt.savefig(f"{file_name}/{label}-times-f1.pdf", format='pdf')

def plot_csv_metrics(data, file_name, label):

    bar_width = 0.35

    prec_color = 'black'
    rec_color = 'red'
    f1_edges_color = 'red'
    f1_nodes_color = 'black'

    ax_limits = [0, 1]

    # Prepare data for plotting
    exercises = list(data.keys())
    edges_precision_avg = [_calculate_average(data[ex]['edges_precision']) for ex in exercises]
    edges_recall_avg = [_calculate_average(data[ex]['edges_recall']) for ex in exercises]
    nodes_precision_avg = [_calculate_average(data[ex]['nodes_precision']) for ex in exercises]
    nodes_recall_avg = [_calculate_average(data[ex]['nodes_recall']) for ex in exercises]
    edges_f1 = [list(map(float, data[ex]['edges_f1'])) for ex in exercises]
    nodes_f1 = [list(map(float, data[ex]['nodes_f1'])) for ex in exercises]

    ex_indexes = exercises
    index = np.arange(len(exercises))

    # Plot 1: Avg Precision and Recall for Edges

    _plot_avg(index, bar_width, edges_precision_avg, edges_recall_avg, prec_color, rec_color, 'Precision', 'Recall',
             'Exercise', 'Average Score', 'Average Precision and Recall for Edges', ex_indexes,
             ax_limits, f"{file_name}/{label}-graph-precision_recall_edges.pdf")

    # Plot 2: Avg Precision and Recall for Nodes

    _plot_avg(index, bar_width, nodes_precision_avg, nodes_recall_avg, prec_color, rec_color, 'Precision', 'Recall',
             'Exercise', 'Average Score', 'Average Precision and Recall for Nodes', ex_indexes,
             ax_limits, f"{file_name}/{label}-graph-precision_recall_nodes.pdf")

    # Calculate average F1 scores for the line chart
    edges_f1_avg = [_calculate_average(data[ex]['edges_f1']) for ex in exercises]
    nodes_f1_avg = [_calculate_average(data[ex]['nodes_f1']) for ex in exercises]

    # Plot 3: Line chart of Avg F1 Scores for Edges and Nodes

    _plot_line_chart(ex_indexes, edges_f1_avg, nodes_f1_avg, f1_edges_color, f1_nodes_color, 'Avg F1 of Edges',
                    'Avg F1 of Nodes', 'Exercise', 'Average F1 Score', 'Average F1 Score for Edges and Nodes',
                    ax_limits, f"{file_name}/{label}-graph-f1_scores_edges_nodes.pdf")

    # Plot 4: Boxplot of F1 Measure for Edges

    _plot_boxplot(edges_f1, ex_indexes, 'Exercise', 'F1 Score', 'Boxplot of F1 Measure for Edges',
                  ax_limits, f"{file_name}/{label}-graph-boxplot_f1_edges.pdf")

    # Plot 5: Boxplot of F1 Measure for Nodes

    _plot_boxplot(nodes_f1, ex_indexes, 'Exercise', 'F1 Score', 'Boxplot of F1 Measure for Nodes',
                  ax_limits, f"{file_name}/{label}-graph-boxplot_f1_nodes.pdf")
