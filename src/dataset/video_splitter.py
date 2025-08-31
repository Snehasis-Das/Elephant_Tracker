import os
import subprocess
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

RAW_DIR = "data/raw_videos"
OUT_DIR = "data/video_data"

os.makedirs(OUT_DIR, exist_ok=True)

def split_video(input_path, output_dir, clip_counter):
    # Load video
    video = open_video(input_path)

    # Setup scene manager
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))  # adjust sensitivity

    # Detect scenes
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    print(f"Found {len(scene_list)} scenes in {os.path.basename(input_path)}")

    if not scene_list:
        return clip_counter

    # Loop over scenes and cut them with ffmpeg
    for start, end in scene_list:
        clip_counter += 1
        out_name = f"clip_{clip_counter:04d}.mp4"
        out_path = os.path.join(output_dir, out_name)

        start_time = start.get_seconds()
        duration = end.get_seconds() - start_time

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c", "copy",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"Saved {out_name} ({duration:.2f} sec)")

    return clip_counter

def create_clip_from_raw():
    clip_counter = 0
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith(".mp4"):
            input_path = os.path.join(RAW_DIR, filename)
            print(f"Processing {filename}...")
            clip_counter = split_video(input_path, OUT_DIR, clip_counter)

if __name__ == "__main__":
    create_clip_from_raw()