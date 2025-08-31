import os
import cv2
from tqdm import tqdm
from .detector import ElephantDetector
from .tracker import ElephantTracker

# Directories
VIDEO_DIR = "data/video_data"
OUT_DIR = "data/detections"

def process_video(video_path, detector, tracker):
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    best_crops = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detector.detect(frame)

        # DeepSORT expects detections -> (tlbr, conf, cls)
        tracks = tracker.update(detections, frame)

        for track in tracks:
            track_id, ltrb, conf, cls_name = track  # unpack tuple
            x1, y1, x2, y2 = map(int, ltrb)

            crop = frame[y1:y2, x1:x2]

            prev = best_crops.get(track_id)
            if prev is None or (conf is not None and conf > prev["conf"]):
                fname = f"{clip_name}_{track_id}.png"
                out_path = os.path.join(OUT_DIR, fname)
                cv2.imwrite(out_path, crop)

                best_crops[track_id] = {
                    "file": fname,
                    "conf": float(conf) if conf else 0.0,
                    "class": cls_name,
                }

    cap.release()

def detect_elephants():
    os.makedirs(OUT_DIR, exist_ok=True)

    detector = ElephantDetector(model_name="yolov8n.pt")
    tracker = ElephantTracker()

    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    for video_file in tqdm(videos, desc="Processing videos"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        print(f"Processing {video_file}...")
        process_video(video_path, detector, tracker)


if __name__ == "__main__":
    detect_elephants()