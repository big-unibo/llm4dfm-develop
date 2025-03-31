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

    ax_limits = [0, 1]
    # Compute average F1-score and elapsed time for each exercise

    exercise_stats = data.groupby("ex_name").agg({
        "edges_f1": "mean",
        "nodes_f1": "mean",
        "time": "mean"
    }).reset_index()

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))

    # Scatter plot for edges
    plt.scatter(exercise_stats["edges_f1"], exercise_stats["time"],
                color=f1_edges_color, label="Edges", alpha=0.7)

    # Scatter plot for nodes
    plt.scatter(exercise_stats["nodes_f1"], exercise_stats["time"],
                color=f1_nodes_color, label="Nodes", alpha=0.7)

    # Annotate each point with exercise name
    for i, row in exercise_stats.iterrows():
        plt.text(row["edges_f1"], row["time"], row["ex_name"],
                 fontsize=6, ha="right", color=f1_edges_color, rotation=45)
        plt.text(row["nodes_f1"], row["time"], row["ex_name"],
                 fontsize=6, ha="right", color=f1_nodes_color, rotation=45)

    # Labels and title
    plt.xlabel("Average F1-score")
    plt.ylabel("Average Elapsed Time (s)")
    plt.title(f"F1-score vs. Elapsed Time [{data.at[0, 'config_label']}]")
    plt.xlim(ax_limits)
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
    
    # # Calculate average F1 scores for the line chart
    # edges_f1_avg = [calculate_average(data[ex]['edges_f1']) for ex in exercises]
    # nodes_f1_avg = [calculate_average(data[ex]['nodes_f1']) for ex in exercises]
    
    # # Plot 3: Line chart of Avg F1 Scores for Edges and Nodes
    # axs[0, 2].plot(ex_indexes, edges_f1_avg, color=f1_edges_color, label='Avg F1 of Edges')
    # axs[0, 2].plot(ex_indexes, nodes_f1_avg, color=f1_nodes_color, label='Avg F1 of Nodes')
    # axs[0, 2].set_xlabel('Exercise')
    # axs[0, 2].set_ylabel('Average F1 Score')
    # axs[0, 2].set_title('Average F1 Score for Edges and Nodes')
    # axs[0, 2].set_ylim(ax_limits)
    # axs[0, 2].legend()
    
    # # Plot 4: Boxplot of F1 Measure for Edges
    # axs[1, 0].boxplot(edges_f1, labels=ex_indexes)
    # axs[1, 0].set_xlabel('Exercise')
    # axs[1, 0].set_ylabel('F1 Score')
    # axs[1, 0].set_ylim(ax_limits)
    # axs[1, 0].set_title('Boxplot of F1 Measure for Edges')
    
    # # Plot 5: Boxplot of F1 Measure for Nodes
    # axs[1, 1].boxplot(nodes_f1, labels=ex_indexes)
    # axs[1, 1].set_xlabel('Exercise')
    # axs[1, 1].set_ylabel('F1 Score')
    # axs[1, 1].set_ylim(ax_limits)
    # axs[1, 1].set_title('Boxplot of F1 Measure for Nodes')
    
    # # Hide the empty subplot (bottom right)
    # axs[1, 2].axis('off')
    # plt.savefig(f"{file_name}.pdf", format='pdf')
    
    # # Adjust layout
    # plt.tight_layout()
    # plt.show()
