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

config_order = (
    df.groupby("config_label")["avg_f1"]
      .mean()
      .sort_values(ascending=False)
      .index
)

# 2. Build pair order: for each config, list cpu first, then gpu
devices = ["cpu", "gpu"]
order = [(cl, dev) for cl in config_order for dev in devices if (cl, dev) in df.set_index(["config_label","device"]).index]

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

# plt.figure(figsize=(12, 6))
#
# # Get categories in order
# categories = df["x_label"].cat.categories
# positions = range(1, len(categories) * 2, 2)  # leave a gap of 1 unit between boxes
#
# # Draw boxplots manually
# data = [df.loc[df["x_label"] == cat, "time"] for cat in categories]
# plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
#
# # Replace x ticks with categories
# plt.xticks(positions, categories, ha="right")
#
# plt.xlabel("Model")
# plt.ylabel("Time")
# plt.title("Execution Times by Model and Device (sorted by avg F1)")
# plt.tight_layout()
#
# plt.savefig(output_file, dpi=300)
# plt.close()
# print(f"Saved boxplot to {output_file}.")

plt.figure(figsize=(12, 6))

# Get categories in order
categories = df["x_label"].cat.categories
positions = range(1, len(categories) * 2, 2)  # leave a gap of 1 unit between boxes

# Draw boxplots manually
data = [df.loc[df["x_label"] == cat, "time"] for cat in categories]
plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

# Now: instead of full labels, split into model+F1 (centered) and device (under each box)

# Build device labels (cpu/gpu)
device_labels = ["CPU" if "cpu" in cat else "GPU" for cat in categories]

# Build F1 labels (one per config_label)
f1_labels = stats.set_index("config_label")["label_text"].to_dict()

# Midpoints for each config (between CPU/GPU positions)
midpoints = []
f1_texts = []
for cl in config_order:
    pair_cats = [label_map[(cl, dev)] for dev in devices if (cl, dev) in label_map]
    pair_positions = [positions[categories.get_loc(cat)] for cat in pair_cats]
    if len(pair_positions) == 2:  # we have cpu+gpu
        mid = sum(pair_positions) / 2
        midpoints.append(mid)
        f1_texts.append(f1_labels[cl])

# First row of ticks: device labels under each box
plt.xticks(positions, device_labels, ha="center")

# Add second row of ticks (F1 labels centered above CPU/GPU pair)
for mid, text in zip(midpoints, f1_texts):
    plt.text(mid, plt.ylim()[0] - 0.05*(plt.ylim()[1]-plt.ylim()[0]),
             text, ha="center", va="top", fontsize=9)

plt.xlabel("Model (with average F1)", labelpad=40)
plt.ylabel("Time")
plt.title("Execution Times by Model and Device (sorted by avg F1)")
plt.tight_layout()

plt.savefig(output_file, dpi=300)
plt.close()