import cv2, os
from pathlib import Path

def save_frames_at_fps(video_path, output_dir, target_fps):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0  # fallback
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

def process_folder_real(root="gen_ai", target_fps=4):
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"Folder not found: {root_p.resolve()}")

    # Collect files named like 1.mp4, 2.mp4, ... and sort numerically by stem
    videos = []
    for p in root_p.iterdir():
        if p.is_file() and p.suffix.lower() == ".mp4":
            try:
                num = int(p.stem)  # only numeric names
                videos.append((num, p))
            except ValueError:
                pass
    videos.sort(key=lambda x: x[0])  # sort by numeric index

    for num, vid_path in videos:
        out_dir = root_p / str(num)  # e.g., real/1, real/2, ...
        print(f"Processing {vid_path} -> {out_dir}")
        save_frames_at_fps(vid_path, str(out_dir), target_fps)

if __name__ == "__main__":
    process_folder_real(root="ai", target_fps=4)
