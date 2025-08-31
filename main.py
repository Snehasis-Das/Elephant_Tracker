from .src.dataset.video_splitter import create_clip_from_raw
from .src.detection.processor import detect_elephants
from .src.features.extractor import extract_features
from .src.matching.matcher import run_matching

def execute():
    create_clip_from_raw()
    detect_elephants()
    extract_features()
    run_matching()

if __name__ == "__main__":
    execute()