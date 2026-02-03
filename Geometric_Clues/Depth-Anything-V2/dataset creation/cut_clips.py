# save as cut_clips.py
import os
import subprocess
import json
from pathlib import Path

FFPROBE = "ffprobe"  # must be on PATH
FFMPEG = "ffmpeg"    # must be on PATH

def get_duration(path):
    # returns duration in seconds (float), or None on failure
    try:
        cmd = [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "json", str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        dur = data.get("format", {}).get("duration")
        return float(dur) if dur is not None else None
    except Exception:
        return None

def process_video(path, out_dir, min_keep=5.0, clip_len=8.0):
    dur = get_duration(path)
    if dur is None:
        print(f"Skip (no duration): {path}")
        return
    if dur < min_keep:
        print(f"Delete (<{min_keep}s): {path}")
        try:
            path.unlink()
        except Exception as e:
            print(f"Failed to delete {path}: {e}")
        return
    # Keep only the first 8s. Use -t 8 to limit duration.
    out_name = path.stem + "_8s" + path.suffix
    out_path = out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # If you want stream copy (fast) but may cut on keyframes:
    # cmd = [FFMPEG, "-y", "-i", str(path), "-t", str(clip_len), "-c", "copy", str(out_path)]
    # For accurate 8s regardless of keyframes, re-encode:
    cmd = [
        FFMPEG, "-y", "-i", str(path),
        "-t", str(clip_len),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path)
    ]
    print("Trimming:", path, "->", out_path)
    subprocess.run(cmd, check=True)

def batch_process(root="real", out_root="real_8s", min_keep=5.0, clip_len=8.0):
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".m4v", ".webm"}
    root_p = Path(root)
    out_root_p = Path(out_root)
    for p in sorted(root_p.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in video_exts:
            process_video(p, out_root_p, min_keep=min_keep, clip_len=clip_len)

if __name__ == "__main__":
    # Input videos in folder "real"
    # Output trimmed clips go to "real_8s"
    batch_process(root="videos", out_root="gen_ai_new", min_keep=5.0, clip_len=8.0)
