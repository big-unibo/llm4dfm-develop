import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from dotenv import load_dotenv

def collect_csv_data(root_dir: str):
    """
    Traverse directories under root_dir recursively,
    reading CSVs only if in the filename 'gpt4' or 'gpt5' appears.
    Returns a DataFrame with columns: config_label, ex_prompt_version, time, edges_f1, nodes_f1, model.
    """
    rows = []
    for subdir, _, files in os.walk(root_dir):
        subdir_lower = subdir.lower()
        if "gpt4" not in subdir_lower and "gpt5" not in subdir_lower:
            continue

        for file in files:
            if not file.endswith(".csv") or ("gpt4" not in file and "gpt5" not in file):
                continue

            filepath = os.path.join(subdir, file)
            model = "gpt4" if "gpt4" in file.lower() else "gpt5"

            try:
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    rows.append({
                        "config_label": row.get("config_label", "unknown"),
                        "ex_prompt_version": row.get("ex_prompt_version", "unknown"),
                        "time": float(row["time"]),
                        "edges_f1": float(row["edges_f1"]),
                        "nodes_f1": float(row["nodes_f1"]),
                        "model": model,
                    })
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    return pd.DataFrame(rows)

load_dotenv()

parser = argparse.ArgumentParser(description="Generate avg F1 vs avg time plot for GPT models.")
parser.add_argument("--root", help="Root directory to scan for CSV files")
args = parser.parse_args()

if args.root:
    start_dir_path = args.root
else:
    start_dir_path = os.getenv('RESULTS')

df = collect_csv_data(start_dir_path)

if df.empty:
    print("No data available to plot.")
    exit(0)

# Compute averages per model and ex_prompt_version
stats = (
    df.groupby(["model", "ex_prompt_version"]).agg(
        avg_time=("time", "mean"),
        avg_edges_f1=("edges_f1", "mean"),
        avg_nodes_f1=("nodes_f1", "mean")
    ).reset_index()
)

# Plot in a single figure
plt.figure(figsize=(10, 6))
markers = {"edges": "o", "nodes": "x"}
colors = {"gpt4": "blue", "gpt5": "red"}

for _, row in stats.iterrows():
    # Scatter points for edges and nodes
    plt.scatter(
        row["avg_time"],
        row["avg_edges_f1"],
        color=colors[row["model"]],
        marker=markers["edges"],
        s=80
    )
    plt.scatter(
        row["avg_time"],
        row["avg_nodes_f1"],
        color=colors[row["model"]],
        marker=markers["nodes"],
        s=80
    )

    # Annotate points with model + prompt version
    plt.text(
        row["avg_time"] + 0.01, row["avg_edges_f1"] + 0.01,
        f"{row['model']} - {row['ex_prompt_version']} (E)", fontsize=8
    )
    plt.text(
        row["avg_time"] + 0.01, row["avg_nodes_f1"] - 0.02,
        f"{row['model']} - {row['ex_prompt_version']} (N)", fontsize=8
    )

    # Draw line between edges and nodes points for same model+prompt
    plt.plot(
        [row["avg_time"], row["avg_time"]],
        [row["avg_edges_f1"], row["avg_nodes_f1"]],
        color=colors[row["model"]], linestyle="--", alpha=0.7
    )

plt.xlabel("Average Execution Time")
plt.ylabel("Average F1 Score")
plt.title("Average F1 vs Average Time by Model and Prompt Version")
plt.xlim(left=0)
plt.ylim(0, 1.1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

output_file = os.path.join(start_dir_path, "aggregate_times_gpts.pdf")
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Saved plot to {output_file}.")
