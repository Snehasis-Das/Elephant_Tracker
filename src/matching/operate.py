# src/matching/operate.py
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tabulate import tabulate


def load_features(feature_dir="data/features"):
    """Load features and keep track of metadata (filename, video_id, frame)."""
    files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".npy")])
    if not files:
        raise RuntimeError(f"No .npy features found in {feature_dir}")

    features, meta = [], []

    for f in files:
        path = os.path.join(feature_dir, f)
        vec = np.load(path)

        # Extract metadata (assuming filename like clip_0001_5.npy)
        base = os.path.splitext(f)[0]
        parts = base.split("_")
        clip_id = parts[1] if len(parts) > 1 else "unknown"
        frame_id = parts[-1] if len(parts) > 1 else "0"

        features.append(vec)
        meta.append({"file": f, "clip": clip_id, "frame": frame_id})

    return np.array(features), pd.DataFrame(meta)


def cluster_features(features, eps=0.1, min_samples=2):
    """Cluster feature vectors using DBSCAN (cosine metric)."""
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine"
    ).fit(features)
    return clustering.labels_


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

    print(f"[Operate] Reports saved in {results_dir}")


def run_matching(feature_dir="data/features", results_dir="data/results"):
    """Main entrypoint for clustering + reporting."""
    print("[Operate] Loading features...")
    features, meta = load_features(feature_dir)

    print("[Operate] Clustering features...")
    labels = cluster_features(features)

    meta["cluster"] = labels

    print("[Operate] Generating reports...")
    generate_reports(meta, results_dir=results_dir)

    return meta


if __name__ == "__main__":
    run_matching()
