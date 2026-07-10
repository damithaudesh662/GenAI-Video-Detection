import os
import subprocess
import json
from pathlib import Path
import cv2
import tempfile
import shutil

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

# -----------------------------
# Get video duration
# -----------------------------
def get_duration(path):
    try:
        cmd = [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "json", str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        dur = data.get("format", {}).get("duration")
        return float(dur) if dur else None
    except Exception:
        return None

# -----------------------------
# Get video FPS
# -----------------------------
def get_fps(path):
    try:
        cmd = [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        num, denom = map(int, out.decode().strip().split("/"))
        return num / denom
    except Exception:
        return None

# -----------------------------
# Trim & optionally downsample to 24 FPS (temporary file)
# -----------------------------
def trim_and_adjust_fps(path, clip_len=5.0, target_fps=24):
    dur = get_duration(path)
    if dur is None or dur < clip_len:
        print(f"Skipping (too short or no duration): {path}")
        return None

    src_fps = get_fps(path)
    if src_fps is None:
        src_fps = 30.0  # fallback

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    # Only change FPS if original FPS != target
    fps_args = ["-r", str(target_fps)] if abs(src_fps - target_fps) > 0.01 else []

    cmd = [
        FFMPEG, "-y", "-i", str(path),
        "-t", str(clip_len),
        *fps_args,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(temp_path)
    ]
    subprocess.run(cmd, check=True)
    return temp_path

# -----------------------------
# Extract frames at 24 FPS
# -----------------------------
def save_frames_at_fps(video_path, output_dir, target_fps=24):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    out_idx = 0
    next_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / src_fps
        if t + 1e-9 >= next_time:
            cv2.imwrite(os.path.join(output_dir, f"frame_{out_idx:05d}.jpg"), frame)
            out_idx += 1
            next_time += 1.0 / target_fps
        frame_idx += 1
    cap.release()

# -----------------------------
# Batch process folder
# -----------------------------
def process_videos_folder(input_folder, output_folder, clip_len=5.0, target_fps=24):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".m4v", ".webm"}
    videos = sorted([p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in video_exts], key=lambda x: x.name)

    # Folder counter (1, 2, 3…)
    for idx, vid_path in enumerate(videos, start=1):
        folder_dir = output_folder / str(idx)
        folder_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing video {idx}: {vid_path.name}")

        # Trim and adjust FPS (temporary video)
        temp_video = trim_and_adjust_fps(vid_path, clip_len=clip_len, target_fps=target_fps)
        if temp_video is None:
            continue

        # Extract frames
        save_frames_at_fps(temp_video, folder_dir, target_fps=target_fps)

        # Delete temporary video
        temp_video.unlink()
        print(f"Saved frames to {folder_dir}\n")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    process_videos_folder(
        input_folder=Path(r"C:\Users\Admin\OneDrive\Desktop\my acadamics\sem 7\research\Git\GenAIVideoDetection\Geometric_Clues\Depth-Anything-V2\dataset\new"),         # folder containing all videos
        output_folder="gen_ai_frames", # folder to save frames
        clip_len=5.0,
        target_fps=5
    )