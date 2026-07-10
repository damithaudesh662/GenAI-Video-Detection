import os
import sys
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor

# Add the project root to sys.path to import existing modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Import existing functional components
from gen_ai_detector.process_video import process_video
from depth_anything_v2.dpt import DepthAnythingV2
from gen_ai_detector.explainability_toolkit import generate_layered_report

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Depth Anything V2 Batch Config
DEPTH_BATCH_SIZE = 8

model_configs = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

# --------------------------------------------------
# Optimized Depth Generation
# --------------------------------------------------
@torch.no_grad()
def optimized_make_depthmaps(model, input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not frame_paths: return

    def load_and_preprocess(path):
        img = cv2.imread(str(path))
        orig_shape = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_resized = cv2.resize(img_rgb, (518, 518)) # DepthAnything input
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        return img_tensor, orig_shape, path.stem

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_and_preprocess, frame_paths))

    tensors = torch.stack([r[0] for r in results]).to(DEVICE)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    tensors = (tensors - mean) / std

    depth_out = []
    for i in range(0, len(tensors), DEPTH_BATCH_SIZE):
        batch = tensors[i:i+DEPTH_BATCH_SIZE]
        depth_out.append(model(batch))
    
    depth_maps = torch.cat(depth_out, dim=0)

    def save_depth(idx):
        depth = depth_maps[idx].cpu().numpy()
        h, w = results[idx][1]
        name = results[idx][2]
        depth_res = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        d_min, d_max = depth_res.min(), depth_res.max()
        depth_norm = ((depth_res - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        color_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(output_dir / f"{name}_depth.png"), color_depth)

    with ThreadPoolExecutor() as executor:
        list(executor.map(save_depth, range(len(depth_maps))))

# --------------------------------------------------
# Global Model Instances (Lazy Loading)
# --------------------------------------------------
_depth_model = None
_classifier_model = None

def get_depth_model():
    global _depth_model
    if _depth_model is None:
        encoder = "vits"
        _depth_model = DepthAnythingV2(**model_configs[encoder])
        _depth_model.load_state_dict(torch.load(ROOT/f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"))
        _depth_model = _depth_model.to(DEVICE).eval()
    return _depth_model

def get_classifier_model(model_path):
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = torchvision.models.video.r3d_18(weights=None)
        _classifier_model.fc = nn.Linear(_classifier_model.fc.in_features, 2)
        _classifier_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        _classifier_model = _classifier_model.to(DEVICE).eval()
    return _classifier_model

# --------------------------------------------------
# Main Backend Functions
# --------------------------------------------------

def process_single_video(video_path, model_path, output_dir):
    video_path, output_dir = Path(video_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / f"temp_{video_path.stem}"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        # 1. Process video (Extract frames)
        process_video(video_path, t=5, fps=4, output_root=temp_dir)
        frame_dir, depth_dir = temp_dir / "frames", temp_dir / "depthmaps"

        # 2. Optimized Depth Generation
        depth_model = get_depth_model()
        optimized_make_depthmaps(depth_model, frame_dir, depth_dir)

        # 3. Inference + layered explainability (delegated to explainability_toolkit)
        classifier = get_classifier_model(model_path)
        report = generate_layered_report(
            depth_dir=str(depth_dir),
            rgb_dir=str(frame_dir),
            output_dir=str(output_dir),
            model=classifier,
            video_name=f"result_{video_path.stem}",
        )

        # 'heatmap_path' kept for backward compatibility with app.py
        return {
            "video":              video_path.name,
            "prediction":         report["prediction"],
            "confidence":         report["confidence"],
            "heatmap_path":       report["summary_panel_path"],
            "temporal_grid_path": report["temporal_grid_path"],
            "temporal_gif_path":  report["temporal_gif_path"],
            "prob_genai":         report["prob_genai"],
            "prob_real":          report["prob_real"],
            "explanation":        report["explanation"],
            "features":           report["features"],
        }
    except Exception as e:
        print(f"Error processing {video_path.name}: {e}"); return {"video": video_path.name, "error": str(e)}
    finally:
        if temp_dir.exists(): shutil.rmtree(temp_dir)

def process_batch_videos(video_folder, model_path, output_root):
    video_folder, output_root = Path(video_folder), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_files = [p for p in video_folder.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}]
    
    # Pre-load models once for the batch
    get_depth_model()
    get_classifier_model(model_path)
    
    return [process_single_video(vid, model_path, output_root) for vid in video_files]

# Example Usage:
# if __name__ == "__main__":
#     model = "../best_r3d18_depthmaps_full.pt"
#     # Single
#     # print(process_single_video("test.mp4", model, "results"))
#     # Batch
#     # print(process_batch_videos("input_folder", model, "batch_results"))
