"""
Joined dataset-build pipeline
  Stage 1 – trim raw video to CLIP_LEN_S seconds       (ffmpeg, temp file)
  Stage 2 – extract frames at TARGET_FPS               (cv2,    temp dir)
  Stage 3 – generate float16 depth maps                (GPU batched)

Input  : dataset/real/*.mp4  |  dataset/gen_ai/*.mp4  (any common video ext)
Output : dataset/depthmaps/real/<video_stem>/frame_NNNNN_depth.npz
         dataset/depthmaps/gen_ai/<video_stem>/frame_NNNNN_depth.npz

Trimmed clips and extracted frames are kept in a TemporaryDirectory and
deleted automatically when each video finishes (or errors out).
Re-runs skip videos whose output folder already contains .npz files.
Partial output from a failed run is removed so the video is retried cleanly.
"""

import cv2
import json
import shutil
import subprocess
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from depth_anything_v2.dpt import DepthAnythingV2
from config import CLIP_LEN_S, TARGET_FPS

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
print(f"Using device: {DEVICE}")

# ── Depth model ───────────────────────────────────────────────────────────────
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

encoder    = 'vits'   # 'vitb' / 'vitl' / 'vitg' for higher quality at cost of speed
BATCH_SIZE = 8        # tune to VRAM: 4 for <6 GB, 8 for 8 GB, 16 for 12+ GB
INPUT_SIZE = 518      # Depth-Anything-V2 default input resolution

model = DepthAnythingV2(**model_configs[encoder])
state = torch.load(
    f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True
)
model.load_state_dict(state)
model = model.to(DEVICE).eval()

# ── Pipeline parameters ───────────────────────────────────────────────────────
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.flv'}
MIN_KEEP_S = 5.0    # skip videos shorter than this (seconds)
# CLIP_LEN_S and TARGET_FPS come from config.py
FFPROBE    = 'ffprobe'
FFMPEG     = 'ffmpeg'


# ── Stage 1 helpers ───────────────────────────────────────────────────────────
def _get_duration(video_path):
    try:
        cmd = [
            FFPROBE, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'format=duration',
            '-of', 'json', str(video_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        dur = json.loads(out.decode('utf-8', errors='ignore')).get('format', {}).get('duration')
        return float(dur) if dur is not None else None
    except Exception:
        return None


def _trim_video(src, dst, clip_len):
    subprocess.run(
        [
            FFMPEG, '-y', '-i', str(src),
            '-t', str(clip_len),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _extract_frames(video_path, out_dir, target_fps):
    """Write JPEG frames; returns number of frames written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = out_idx = 0
    next_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx / src_fps + 1e-9 >= next_time:
            cv2.imwrite(str(out_dir / f'frame_{out_idx:05d}.jpg'), frame)
            out_idx += 1
            next_time += 1.0 / target_fps
        frame_idx += 1

    cap.release()
    return out_idx


# ── Stage 3: batched GPU depth inference ─────────────────────────────────────
def _read_img(path):
    return cv2.imread(str(path))


@torch.inference_mode()
def _generate_depthmaps(img_paths, out_paths):
    """Batch-infer depth; save float16 .npz normalised to [0, 1] per frame."""
    # Parallel disk reads
    with ThreadPoolExecutor(max_workers=4) as ex:
        raw_imgs = list(ex.map(_read_img, img_paths))

    # Sequential preprocessing (model.image2tensor calls torch on DEVICE)
    tensors, orig_sizes, valid_idx = [], [], []
    for i, img in enumerate(raw_imgs):
        if img is None:
            print(f"    skip unreadable: {img_paths[i].name}")
            continue
        tensor, (h, w) = model.image2tensor(img, INPUT_SIZE)
        tensors.append(tensor.squeeze(0))
        orig_sizes.append((h, w))
        valid_idx.append(i)

    if not tensors:
        return

    # Batched GPU forward pass
    for start in range(0, len(tensors), BATCH_SIZE):
        chunk_t  = tensors[start : start + BATCH_SIZE]
        chunk_sz = orig_sizes[start : start + BATCH_SIZE]
        chunk_vi = valid_idx[start : start + BATCH_SIZE]

        batch = torch.stack(chunk_t)  # already on DEVICE

        if DEVICE == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                depths = model(batch)
        else:
            depths = model(batch)

        # All frames from the same video share the same original resolution
        h, w = chunk_sz[0]
        depths_up = F.interpolate(
            depths[:, None].float(), (h, w), mode='bilinear', align_corners=True
        )[:, 0]  # (B, H, W)

        for j, i in enumerate(chunk_vi):
            depth = depths_up[j].cpu().numpy()
            d_min, d_max = float(depth.min()), float(depth.max())
            if d_max - d_min > 1e-8:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = np.zeros_like(depth)

            out_path = out_paths[i]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out_path), depth=depth.astype(np.float16))
            print(f"    {img_paths[i].name} -> {out_path.name}")


# ── Joined pipeline ───────────────────────────────────────────────────────────
def build_dataset_depthmaps(dataset_root='dataset'):
    root       = Path(dataset_root)
    depth_root = root / 'depthmaps'

    for src in [root / 'real', root / 'gen_ai']:
        if not src.exists():
            continue

        videos = sorted(
            [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
            key=lambda p: p.name,
        )

        if not videos:
            print(f"[{src.name}] no video files found, skipping.")
            continue

        for video_path in videos:
            out_dir = depth_root / src.name / video_path.stem

            if out_dir.exists() and any(out_dir.glob('*.npz')):
                print(f"[{src.name}/{video_path.name}] already done, skipping.")
                continue

            dur = _get_duration(video_path)
            if dur is None:
                print(f"[{src.name}/{video_path.name}] cannot read duration, skipping.")
                continue
            if dur < MIN_KEEP_S:
                print(f"[{src.name}/{video_path.name}] too short ({dur:.1f}s < {MIN_KEEP_S}s), skipping.")
                continue

            print(f"\n[{src.name}/{video_path.name}] {dur:.1f}s")

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path  = Path(tmp)
                trimmed   = tmp_path / f'{video_path.stem}_trimmed.mp4'
                frame_dir = tmp_path / 'frames'

                try:
                    # Stage 1 — trim
                    print(f"  [1/3] trimming to {CLIP_LEN_S}s ...")
                    _trim_video(video_path, trimmed, CLIP_LEN_S)

                    # Stage 2 — extract frames, then immediately free the trimmed clip
                    print(f"  [2/3] extracting frames at {TARGET_FPS} FPS ...")
                    n = _extract_frames(trimmed, frame_dir, TARGET_FPS)
                    trimmed.unlink()   # trimmed clip no longer needed
                    print(f"        {n} frames extracted.")

                    # Stage 3 — depth maps
                    img_paths = sorted(frame_dir.iterdir(), key=lambda p: p.name)
                    out_paths = [out_dir / (p.stem + '_depth.npz') for p in img_paths]
                    print(f"  [3/3] generating {len(img_paths)} depth maps -> {out_dir}")
                    _generate_depthmaps(img_paths, out_paths)

                    # frame_dir is deleted when TemporaryDirectory exits
                    print(f"  done.")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    # Remove partial output so re-run retries this video from scratch
                    if out_dir.exists():
                        shutil.rmtree(out_dir)


if __name__ == '__main__':
    build_dataset_depthmaps(dataset_root='dataset')
