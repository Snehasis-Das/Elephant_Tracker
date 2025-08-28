import os
import cv2
import numpy as np
from tqdm import tqdm
from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

# Paths
IN_DIR = "data/detections"
OUT_DIR = "data/features"
os.makedirs(OUT_DIR, exist_ok=True)


def ensure_bgr(img):
    """
    Convert any image to BGR with 3 channels.
    Supports grayscale, single-channel, RGBA, or malformed images.
    """
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape) == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:  # single channel with extra dim
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        raise ValueError(f"Unexpected number of channels: {img.shape[2]}")
    return img


class FeatureExtractor:
    def __init__(self):
        """Wrapper around DeepSORT's MobileNetv2 embedder."""
        self.embedder = MobileNetv2_Embedder()

    def extract(self, img):
        """
        Extract embedding from image.
        Input: numpy array (any channels)
        Output: 1D numpy vector
        """
        img_bgr = ensure_bgr(img)
        vecs = self.embedder.predict(img_bgr)

        # Handle batch output
        if isinstance(vecs, (list, tuple)):
            return vecs[0]
        return vecs


def main():
    extractor = FeatureExtractor()

    for fname in tqdm(os.listdir(IN_DIR), desc="Extracting features"):
        if not fname.lower().endswith(".png"):
            continue

        in_path = os.path.join(IN_DIR, fname)
        # Read in unchanged mode to catch grayscale, RGBA, etc.
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Failed to load {fname}")
            continue

        try:
            feat = extractor.extract(img)  # numpy vector
        except Exception as e:
            print(f"Error extracting features from {fname}: {e}")
            continue

        # Save as .npy
        base = fname.replace("clip", "feature").replace(".png", ".npy")
        out_path = os.path.join(OUT_DIR, base)
        np.save(out_path, feat.astype(np.float32))


if __name__ == "__main__":
    main()
