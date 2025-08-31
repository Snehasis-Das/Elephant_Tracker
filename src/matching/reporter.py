# src/matching/reporter.py
import os
import pandas as pd
from tabulate import tabulate

def generate_reports(df, results_dir="data/results"):
    """Save results as CSV + TXT + per-elephant summary."""
    os.makedirs(results_dir, exist_ok=True)

    # Save raw results
    csv_path = os.path.join(results_dir, "clusters.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(results_dir, "clusters.txt")
    with open(txt_path, "w") as f:
        f.write(tabulate(df, headers="keys", tablefmt="pretty"))

    # Per-elephant grouping
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for cluster_id, group in df.groupby("cluster"):
            if cluster_id == -1:
                continue  # skip noise
            f.write(f"\n=== Elephant {cluster_id} ===\n")
            f.write(f"Frames: {len(group)}\n")
            f.write("Clips: " + ", ".join(sorted(group['clip'].unique())) + "\n")
            f.write("Files:\n")
            for file in group["file"]:
                f.write(f"  - {file}\n")

    print(f"[Reporter] Reports saved in {results_dir}")
