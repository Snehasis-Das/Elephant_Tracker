# src/matching/matcher.py
import os
import numpy as np
import pandas as pd

from .clusterer import cluster_features
from .reporter import generate_reports


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


def run_matching(feature_dir="data/features", results_dir="data/results", eps=0.1, min_samples=2):
    """Main entrypoint for clustering + reporting."""
    print("[Matcher] Loading features...")
    features, meta = load_features(feature_dir)

    print("[Matcher] Clustering features...")
    labels = cluster_features(features, eps=eps, min_samples=min_samples)

    meta["cluster"] = labels

    print("[Matcher] Generating reports...")
    generate_reports(meta, results_dir=results_dir)

    return meta


if __name__ == "__main__":
    run_matching()
