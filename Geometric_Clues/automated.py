# save as make_depthmaps.py
import os
import cv2
import torch
import numpy as np
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2

# Choose best device automatically
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

# pick encoder here
encoder = 'vits'  # or 'vitb', 'vitl', 'vitg'

# load model
model = DepthAnythingV2(**model_configs[encoder])
state = torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu')
model.load_state_dict(state)
model = model.to(DEVICE).eval()

# image types
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

def process_image(img_path: Path, out_path: Path):
    raw_img = cv2.imread(str(img_path))
    if raw_img is None:
        print(f"Skip unreadable: {img_path}")
        return
    # infer depth
    depth = model.infer_image(raw_img)  # expected HxW float array
    # normalize to 0-255
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-8:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
    else:
        depth_vis = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
    # colorize
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    # write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), depth_color)

def mirror_depthmaps(dataset_root="dataset"):
    root = Path(dataset_root)
    sources = [root/"real", root/"gen_ai"]
    depth_root = root/"depthmaps"

    for src in sources:
        if not src.exists():
            continue
        # iterate numeric subfolders (1, 2, 3, ...)
        for sub in sorted([d for d in src.iterdir() if d.is_dir()], key=lambda p: p.name):
            # Only process subfolders that are numeric names if desired:
            # if not sub.name.isdigit(): continue
            for img_file in sorted(sub.iterdir(), key=lambda p: p.name):
                if img_file.is_file() and img_file.suffix.lower() in extensions:
                    rel = img_file.relative_to(root)  # e.g., real/1/frame_00001.jpg
                    out_rel = rel  # keep same relative path
                    # change top-level root to depthmaps and filenames to *_depth_color.png
                    out_rel_parts = list(out_rel.parts)
                    out_rel_parts[0] = "depthmaps"  # replace dataset top-level child (real/gen_ai) root later
                    # build output path under dataset/depthmaps/<same tree>
                    out_dir = depth_root / rel.parts[0] / rel.parts[1]
                    base = img_file.stem
                    out_file = base + "_depth_color.png"
                    out_path = depth_root / rel.parts[0] / rel.parts[1] / out_file
                    # ensure correct mirrored structure: dataset/depthmaps/real/1/...
                    out_path = depth_root / src.name / sub.name / out_file

                    print(f"{img_file} -> {out_path}")
                    process_image(img_file, out_path)

if __name__ == "__main__":
    mirror_depthmaps(dataset_root="dataset")
