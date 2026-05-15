# import cv2
# import torch
# import numpy as np
# from pathlib import Path

# from depth_anything_v2.dpt import DepthAnythingV2

# # --------------------------------------------------
# # Device selection
# # --------------------------------------------------
# DEVICE = (
#     "cuda" if torch.cuda.is_available()
#     else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using device: {DEVICE}")

# # --------------------------------------------------
# # Model configuration
# # --------------------------------------------------
# model_configs = {
#     "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
#     "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
#     "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
#     "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
# }

# encoder = "vits"  # choose encoder here

# # --------------------------------------------------
# # Load model
# # --------------------------------------------------
# model = DepthAnythingV2(**model_configs[encoder])
# ckpt_path = f"checkpoints/depth_anything_v2_{encoder}.pth"

# state = torch.load(ckpt_path, map_location="cpu")
# model.load_state_dict(state)
# model = model.to(DEVICE).eval()

# # --------------------------------------------------
# # Image extensions
# # --------------------------------------------------
# EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# # --------------------------------------------------
# # Process one image
# # --------------------------------------------------
# def process_image(img_path: Path, out_path: Path):
#     img = cv2.imread(str(img_path))
#     if img is None:
#         print(f"⚠️  Skipping unreadable file: {img_path}")
#         return

#     # Depth inference
#     depth = model.infer_image(img)  # H x W (float)

#     # Normalize depth to 0–255
#     d_min, d_max = depth.min(), depth.max()
#     if d_max - d_min < 1e-8:
#         depth_norm = np.zeros_like(depth, dtype=np.uint8)
#     else:
#         depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

#     # Optional: colorize (good for visualization)
#     depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path), depth_color)

# # --------------------------------------------------
# # Process one video folder
# # --------------------------------------------------
# def process_video_folder(input_dir: str, output_dir: str):
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)

#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input folder not found: {input_dir}")

#     frames = sorted(
#         [p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS],
#         key=lambda p: p.name
#     )

#     print(f"Found {len(frames)} frames")

#     for frame_path in frames:
#         out_name = frame_path.stem + "_depth.png"
#         out_path = output_dir / out_name

#         print(f"{frame_path.name} -> {out_path.name}")
#         process_image(frame_path, out_path)

# # --------------------------------------------------
# # Main
# # --------------------------------------------------
# if __name__ == "__main__":
#     input_frames_dir = "input_frames/output/4_frames"        # folder with RGB frames
#     output_depth_dir = "output_depthmaps"    # folder to save depth maps

#     process_video_folder(input_frames_dir, output_depth_dir)


import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from depth_anything_v2.dpt import DepthAnythingV2

# --------------------------------------------------
# Device selection
# --------------------------------------------------
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

# --------------------------------------------------
# Model config
# --------------------------------------------------
model_configs = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

encoder    = "vits"
BATCH_SIZE = 8    # tune to VRAM: 4 for <6 GB, 8 for 8 GB, 16 for 12+ GB
INPUT_SIZE = 518

model = DepthAnythingV2(**model_configs[encoder])
state = torch.load(
    ROOT / f"checkpoints/depth_anything_v2_{encoder}.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state)
model = model.to(DEVICE).eval()

# --------------------------------------------------
# Image extensions
# --------------------------------------------------
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_img(path):
    """I/O only — called from thread pool."""
    return cv2.imread(str(path))


@torch.inference_mode()
def _run_batch(img_paths, out_paths):
    """
    Batch-infer depth for img_paths; save float16 compressed .npz to out_paths.
    Depth values are normalised to [0, 1] per frame.
    """
    with ThreadPoolExecutor(max_workers=4) as ex:
        raw_imgs = list(ex.map(_read_img, img_paths))

    tensors, orig_sizes, valid_idx = [], [], []
    for i, img in enumerate(raw_imgs):
        if img is None:
            print(f"  skip unreadable: {img_paths[i].name}")
            continue
        tensor, (h, w) = model.image2tensor(img, INPUT_SIZE)
        tensors.append(tensor.squeeze(0))
        orig_sizes.append((h, w))
        valid_idx.append(i)

    if not tensors:
        return

    for start in range(0, len(tensors), BATCH_SIZE):
        chunk_t  = tensors[start : start + BATCH_SIZE]
        chunk_sz = orig_sizes[start : start + BATCH_SIZE]
        chunk_vi = valid_idx[start : start + BATCH_SIZE]

        batch = torch.stack(chunk_t)

        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                depths = model(batch)
        else:
            depths = model(batch)

        h, w = chunk_sz[0]
        depths_up = F.interpolate(
            depths[:, None].float(), (h, w), mode="bilinear", align_corners=True
        )[:, 0]

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
            print(f"  {img_paths[i].name} -> {out_path.name}")


# --------------------------------------------------
# Process a folder of frames (public API used by extract_logits_depth.py)
# --------------------------------------------------
def make_depthmaps(input_frames_dir, output_depth_dir):
    input_dir  = Path(input_frames_dir)
    output_dir = Path(output_depth_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS],
        key=lambda p: p.name,
    )
    print(f"Found {len(images)} frames")

    img_paths = images
    out_paths = [output_dir / (p.stem + "_depth.npz") for p in images]

    _run_batch(img_paths, out_paths)
    print("Depthmap generation complete.")


# keep single-image helper for any callers that import it directly
def process_image(img_path, out_path):
    _run_batch([Path(img_path)], [Path(out_path)])


# --------------------------------------------------
# CLI
# --------------------------------------------------
# if __name__ == "__main__":
#     """
#     Example:
#     python make_depthmaps.py input_frames output_depthmaps
#     """
#     # import sys

#     # if len(sys.argv) != 3:
#     #     print("Usage: python make_depthmaps.py <input_frames_folder> <output_depthmaps_folder>")
#     #     sys.exit(1)
#     input_frames_dir="output/4_frames"
#     output_depth_dir="output"

#     make_depthmaps(input_frames_dir, output_depth_dir)
