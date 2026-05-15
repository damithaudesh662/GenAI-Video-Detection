import os
import cv2
import json
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# --------------------------------------------------
# Get video duration
# --------------------------------------------------
def get_duration(video_path):
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode())
    return float(data["format"]["duration"])


# --------------------------------------------------
# Cut video to t seconds
# --------------------------------------------------
def cut_video(video_path, out_video, t):
    cmd = [
        FFMPEG, "-y",
        "-i", str(video_path),
        "-t", str(t),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_video)
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------
# Extract frames at target FPS
# --------------------------------------------------
def extract_frames(video_path, out_dir, target_fps):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    frame_idx = 0
    out_idx = 0
    next_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_time = frame_idx / src_fps
        if current_time >= next_time:
            cv2.imwrite(
                str(out_dir / f"frame_{out_idx:05d}.jpg"),
                frame
            )
            out_idx += 1
            next_time += 1.0 / target_fps

        frame_idx += 1

    cap.release()


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def process_video(video_file, t, fps, output_root="depthmaps"):
    video_file = Path(video_file)
    output_root = Path(output_root)

    if not video_file.exists():
        raise FileNotFoundError(video_file)

    duration = get_duration(video_file)
    if duration < t:
        raise ValueError(f"Video shorter than {t}s")

    temp_video = output_root / f"{video_file.stem}_{t}s.mp4"
    frame_dir = output_root / "frames"

    output_root.mkdir(exist_ok=True)

    print("  Cutting video...")
    cut_video(video_file, temp_video, t)

    print("  Extracting frames...")
    extract_frames(temp_video, frame_dir, fps)

    print("  Done!")
    print(f"Frames saved to: {frame_dir.resolve()}")
    
    



# --------------------------------------------------
# CLI usage
# --------------------------------------------------
# if __name__ == "__main__":
#     """
#     Example:
#     python process_video.py input.mp4 8 4
#     """
#     # import sys

#     # if len(sys.argv) != 4:
#     #     print("Usage: python process_video.py <video_file> <t_seconds> <fps>")
#     #     sys.exit(1)

#     # video = sys.argv[1]
#     # t = float(sys.argv[2])
#     # fps = float(sys.argv[3])
#     video="4.mp4"
#     t=5
#     fps=4

#     process_video(video, t, fps)
