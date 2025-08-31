import os
import cv2
import numpy as np
from tqdm import tqdm
from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

IN_DIR = "data/detections"
OUT_DIR = "data/features"
os.makedirs(OUT_DIR, exist_ok=True)


class FeatureExtractor:
    def __init__(self):
        self.embedder = MobileNetv2_Embedder()

    def extract(self, img_bgr):
        # MobileNetv2_Embedder.predict expects a list of images
        vecs = self.embedder.predict([img_bgr])

        if vecs is None or len(vecs) == 0:
            print("⚠️ No feature vector returned")
            return None

        return vecs[0]  # take the first vector


def extract_features():
    extractor = FeatureExtractor()
    files = [f for f in os.listdir(IN_DIR) if f.lower().endswith(".png")]
    print(f"Found {len(files)} images in {IN_DIR}")

    for fname in tqdm(files, desc="Extracting features"):
        in_path = os.path.join(IN_DIR, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"⚠️ Failed to read {in_path}")
            continue

        feat = extractor.extract(img)
        if feat is None:
            print(f"⚠️ No features extracted for {fname}")
            continue

        base = fname.replace("clip", "feature").replace(".png", ".npy")
        out_path = os.path.join(OUT_DIR, base)
        np.save(out_path, feat.astype(np.float32))
        print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    extract_features()
