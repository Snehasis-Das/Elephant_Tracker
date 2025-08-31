# src/matching/reporter.py
import os
import pandas as pd
from tabulate import tabulate

class ReportGenerator:
    def __init__(self, results_dir="data/results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def save_csv(self, df, filename="clusters.csv"):
        path = os.path.join(self.results_dir, filename)
        df.to_csv(path, index=False)
        print(f"[Report] CSV saved to {path}")

    def save_txt(self, df, filename="clusters.txt"):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            f.write(tabulate(df, headers="keys", tablefmt="pretty"))
        print(f"[Report] TXT saved to {path}")

    def summary(self, df):
        counts = df["cluster"].value_counts()
        print("\n[Summary]")
        print(counts)
