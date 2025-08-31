# src/matching/clusterer.py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances

def cluster_features(features, eps=0.1, min_samples=2, diagnostics=True):
    """
    Cluster feature vectors using DBSCAN (cosine metric).
    Returns cluster labels for each feature.
    """
    # Normalize before cosine similarity
    features = normalize(features)

    if diagnostics:
        dists = cosine_distances(features)
        upper = dists[np.triu_indices(len(features), k=1)]
        print(f"[Diagnostics] Cosine dist → min={upper.min():.3f}, "
              f"max={upper.max():.3f}, mean={upper.mean():.3f}")

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine"
    ).fit(features)

    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    print(f"[Clusterer] Found {n_clusters} clusters (+ noise)")
    return clustering.labels_
