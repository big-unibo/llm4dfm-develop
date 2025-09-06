import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from dotenv import load_dotenv

def collect_csv_data(root_dir: str):
    """
    Traverse directories under root_dir recursively,
    reading CSVs only if in the filename 'cpu' or 'gpu' appears.
    Returns a DataFrame with columns: config_label, device, time, edges_f1, nodes_f1.
    """
    rows = []
    for subdir, _, files in os.walk(root_dir):
        # Check full path, not just basename
        subdir_lower = subdir.lower()
        if "cpu" not in subdir_lower and "gpu" not in subdir_lower:
            continue

        device = "cpu" if "cpu" in subdir_lower else "gpu"


        for file in files:
            if not file.endswith(".csv") or ("cpu" not in file and "gpu" not in file):
                continue

            filepath = os.path.join(subdir, file)

            try:
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    rows.append({
                        "config_label": row["config_label"],
                        "device": device,
                        "time": float(row["time"]),
                        "edges_f1": float(row["edges_f1"]),
                        "nodes_f1": float(row["nodes_f1"]),
                    })
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    return pd.DataFrame(rows)

load_dotenv()

parser = argparse.ArgumentParser(description="Generate boxplots of execution times grouped by config and device.")
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

# Compute average F1
df["avg_f1"] = (df["edges_f1"] + df["nodes_f1"]) / 2

# Compute averages per config_label (ignoring device)
stats = (
    df.groupby("config_label")
    .agg(
        avg_f1=("avg_f1", "mean"),
        edges_f1=("edges_f1", "mean"),
        nodes_f1=("nodes_f1", "mean"),
    )
    .reset_index()
)

# Round for readability
stats["avg_f1"] = stats["avg_f1"].round(3)
stats["edges_f1"] = stats["edges_f1"].round(3)
stats["nodes_f1"] = stats["nodes_f1"].round(3)

# Build pretty label (same for CPU/GPU)
stats["label_text"] = (
    stats["config_label"]
    + "\nF1-E=" + stats["edges_f1"].astype(str)
    + "\nF1-N=" + stats["nodes_f1"].astype(str)
)

# Merge back: attach model-level averages to each row
df = df.merge(stats[["config_label", "label_text"]], on="config_label", how="left")

# Final label includes both device and the model-level F1 info
df["x_label"] = df["label_text"] + "\n" + df["device"]

order = (
    df.groupby(["config_label", "device"])["avg_f1"]
      .mean()
      .sort_values(ascending=False)
      .index
)

# Build a mapping (config_label, device) -> enriched label
label_map = df.set_index(["config_label", "device"])["x_label"].to_dict()

# Reorder categorical using the enriched labels
df["x_label"] = pd.Categorical(
    df["x_label"],
    categories=[label_map[(cl, dev)] for cl, dev in order],
    ordered=True
)

output_file = f'{start_dir_path}aggregate_times_cpu_gpu.pdf'

# Plot
plt.figure(figsize=(12, 6))
df.boxplot(column="time", by="x_label", grid=False)
plt.xlabel("Model")
plt.ylabel("Time")
plt.title("Execution Times by Model and Device (sorted by avg F1)")
plt.suptitle("")
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Saved boxplot to {output_file}.")