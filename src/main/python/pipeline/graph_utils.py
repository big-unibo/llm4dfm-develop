import matplotlib.pyplot as plt
import numpy as np


def plot_csv_metrics(data, file_name, label):
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
    fig1.savefig(f"{file_name}/{label}-graph-precision_recall_edges.pdf", format='pdf')

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
    fig2.savefig(f"{file_name}/{label}-graph-precision_recall_nodes.pdf", format='pdf')

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
    fig3.savefig(f"{file_name}/{label}-graph-f1_scores_edges_nodes.pdf", format='pdf')

    # Plot 4: Boxplot of F1 Measure for Edges
    fig4, ax4 = plt.subplots()
    ax4.boxplot(edges_f1, labels=ex_indexes)
    ax4.set_xlabel('Exercise')
    ax4.set_ylabel('F1 Score')
    ax4.set_ylim(ax_limits)
    ax4.set_title('Boxplot of F1 Measure for Edges')
    fig4.tight_layout()
    fig4.savefig(f"{file_name}/{label}-graph-boxplot_f1_edges.pdf", format='pdf')

    # Plot 5: Boxplot of F1 Measure for Nodes
    fig5, ax5 = plt.subplots()
    ax5.boxplot(nodes_f1, labels=ex_indexes)
    ax5.set_xlabel('Exercise')
    ax5.set_ylabel('F1 Score')
    ax5.set_ylim(ax_limits)
    ax5.set_title('Boxplot of F1 Measure for Nodes')
    fig5.tight_layout()
    fig5.savefig(f"{file_name}/{label}-graph-boxplot_f1_nodes.pdf", format='pdf')




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