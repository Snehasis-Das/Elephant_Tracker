# src/matching/matcher.py
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

class ElephantMatcher:
    def __init__(self, feature_dir="data/features", eps=0.5, min_samples=2):
        self.feature_dir = feature_dir
        self.eps = eps
        self.min_samples = min_samples
        self.features = []
        self.meta = []  # filenames

    def load_features(self):
        """Load all features from data/features"""
        files = sorted([f for f in os.listdir(self.feature_dir) if f.endswith(".npy")])
        for f in files:
            path = os.path.join(self.feature_dir, f)
            vec = np.load(path)
            self.features.append(vec)
            self.meta.append(f)
        self.features = np.array(self.features)

    def cluster(self):
        """Cluster features using DBSCAN"""
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="cosine"
        ).fit(self.features)
        labels = clustering.labels_
        return labels

    def run(self):
        self.load_features()
        labels = self.cluster()
        return pd.DataFrame({
            "file": self.meta,
            "cluster": labels
        })