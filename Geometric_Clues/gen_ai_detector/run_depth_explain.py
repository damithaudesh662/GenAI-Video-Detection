#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_depth_explain.py

On-demand Depth / Geometry explainability for a single video.

Pipeline (mirrors backend/processor.py but stripped to one video):
  1. Extract RGB frames with ffmpeg  (via gen_ai_detector.process_video)
  2. Generate colorized depth maps  (Depth Anything V2 ViT-S)
  3. Run GradCAM on R3D-18           (gen_ai_detector.explainability_toolkit)
  4. Write summary_panel.png + result.json

CLI
---
python run_depth_explain.py \\
    --video      /path/to/video.mp4 \\
    --model      /path/to/best_r3d18_depthmaps_full.pt \\
    --output-dir /path/to/output \\
    --output-json /path/to/result.json
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the Depth-Anything-V2 project root is on sys.path
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent   # gen_ai_detector/
_DAV2_ROOT = _HERE.parent                      # Depth-Anything-V2/
if str(_DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(_DAV2_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(message: str, output_json: Optional[str]) -> None:
    payload = json.dumps({"error": message})
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        print(payload, file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Depth / GradCAM on-demand explainability for a single video"
    )
    parser.add_argument("--video",       required=True,
                        help="Path to the input video file")
    parser.add_argument("--model",       required=True,
                        help="Path to R3D-18 .pt classification checkpoint")
    parser.add_argument("--output-dir",  required=True,
                        help="Directory to write summary panel + grid PNG")
    parser.add_argument("--output-json", default=None,
                        help="Path to write result JSON (stdout if omitted)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem

    if not video_path.exists():
        _fail(f"Video not found: {video_path}", args.output_json)
    if not model_path.exists():
        _fail(f"R3D-18 model checkpoint not found: {model_path}", args.output_json)

    # ── Lazy imports (require depth venv) ──────────────────────────────────
    try:
        import torch
        import torch.nn as nn
        import torchvision
        import cv2
        import numpy as np
        from gen_ai_detector.process_video import process_video
        from depth_anything_v2.dpt import DepthAnythingV2
        from gen_ai_detector.explainability_toolkit import generate_layered_report
    except ImportError as exc:
        _fail(f"Import error — run under the depth venv: {exc}", args.output_json)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,
                 "out_channels": [48, 96, 192, 384]},
    }

    # ── Load Depth Anything V2 (ViT-S) ────────────────────────────────────
    depth_ckpt = _DAV2_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"
    if not depth_ckpt.exists():
        _fail(f"Depth Anything V2 checkpoint not found: {depth_ckpt}", args.output_json)

    print(f"[run_depth_explain] Loading Depth Anything V2 from: {depth_ckpt}")
    depth_model = DepthAnythingV2(**model_configs["vits"])
    depth_model.load_state_dict(
        torch.load(str(depth_ckpt), map_location="cpu")
    )
    depth_model = depth_model.to(DEVICE).eval()

    # ── Load R3D-18 classifier ─────────────────────────────────────────────
    print(f"[run_depth_explain] Loading R3D-18 classifier from: {model_path}")
    classifier = torchvision.models.video.r3d_18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    classifier = classifier.to(DEVICE).eval()

    # ── Extract frames + depth maps in a temp dir ──────────────────────────
    DEPTH_BATCH_SIZE = 8
    FRAME_SIZE_DEPTH = 518   # DepthAnythingV2 input resolution

    with tempfile.TemporaryDirectory(prefix="depth_explain_") as tmp:
        tmp_path  = Path(tmp)
        frame_dir = tmp_path / "frames"
        depth_dir = tmp_path / "depthmaps"
        depth_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[run_depth_explain] Extracting frames from: {video_path.name}")
            process_video(video_path, t=5, fps=4, output_root=tmp_path)
        except Exception as exc:
            _fail(f"Frame extraction failed: {exc}", args.output_json)

        # ── Generate depth maps ──
        frame_paths = sorted(
            p for p in frame_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not frame_paths:
            _fail("No frames extracted from video", args.output_json)

        print(f"[run_depth_explain] Generating depth maps for {len(frame_paths)} frames…")

        def _load_frame(path):
            img = cv2.imread(str(path))
            orig_shape = img.shape[:2]
            img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img_res   = cv2.resize(img_rgb, (FRAME_SIZE_DEPTH, FRAME_SIZE_DEPTH))
            tensor    = torch.from_numpy(img_res).permute(2, 0, 1).float()
            return tensor, orig_shape, path.stem

        meta = [_load_frame(p) for p in frame_paths]
        tensors = torch.stack([m[0] for m in meta]).to(DEVICE)

        mean_t = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std_t  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        tensors = (tensors - mean_t) / std_t

        depth_chunks = []
        with torch.no_grad():
            for i in range(0, len(tensors), DEPTH_BATCH_SIZE):
                depth_chunks.append(depth_model(tensors[i:i + DEPTH_BATCH_SIZE]))
        depth_maps = torch.cat(depth_chunks, dim=0)

        for idx, (depth, (h, w), name) in enumerate(zip(depth_maps, [m[1] for m in meta], [m[2] for m in meta])):
            d = depth.cpu().numpy()
            d_res = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            d_min, d_max = d_res.min(), d_res.max()
            d_norm = ((d_res - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
            color_depth = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(depth_dir / f"{name}_depth.png"), color_depth)

        # ── GradCAM + summary panel ──
        print("[run_depth_explain] Running GradCAM explainability…")
        try:
            report = generate_layered_report(
                depth_dir=str(depth_dir),
                rgb_dir=str(frame_dir),
                output_dir=str(output_dir),
                model=classifier,
                video_name=video_stem,
            )
        except Exception as exc:
            _fail(f"generate_layered_report failed: {exc}", args.output_json)

    # ── Build output JSON ──────────────────────────────────────────────────
    summary_filename = Path(report.get("summary_panel_path", "")).name or None
    grid_filename    = Path(report.get("temporal_grid_path", "")).name or None

    output = {
        "modality":          "depth",
        "video_name":        video_stem,
        "prediction":        report.get("prediction", ""),
        "prob_genai":        report.get("prob_genai", 0),
        "prob_real":         report.get("prob_real",  0),
        "confidence":        report.get("confidence", ""),
        "summary_filename":  summary_filename,
        "grid_filename":     grid_filename,
        "explanation":       report.get("explanation", {}),
        "features":          report.get("features",   {}),
    }

    json_str = json.dumps(output, indent=2)
    print(f"[run_depth_explain] Done. Summary panel: {summary_filename}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(json_str)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
