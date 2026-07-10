"""
Explainability Toolkit — GenAI Video Detection
===============================================
Three complementary outputs for every prediction:

  Layer A  – Summary panel      (RGB | Depth | GradCAM overlay + probability bars + bullet text)
  Layer B  – Temporal grid      (4 × 4 PNG, all 16 sampled frames with heatmap overlay)
  Layer C  – Temporal GIF       (animated loop of the 16 heatmap overlays)

Each panel includes both plain-language and technical explanations derived from
three measurable geometric cues: temporal stability, spatial smoothness, and edge density.

Standalone usage:
    python explainability_toolkit.py \\
        --model  ../trained_detection_models/best_r3d18_depthmaps_full.pt \\
        --depth-dir  output/depthmaps \\
        --rgb-dir    output/frames \\
        --output-dir explainability_output \\
        --name       my_video
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

# ── Constants ─────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# R3D-18 normalisation (kinetics pre-training stats)
_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
_STD  = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)

CLIP_LEN   = 16   # frames fed to the classifier
FRAME_SIZE = 112  # spatial resolution expected by R3D-18

# Thresholds that map scalar features to readable language
_TS_LOW,  _TS_HIGH  = 0.015, 0.040   # temporal stability (mean |Δdepth|)
_SM_LOW,  _SM_HIGH  = 0.002, 0.015   # spatial smoothness (Laplacian variance)
_ED_LOW,  _ED_HIGH  = 0.040, 0.120   # edge density (Canny pixel fraction)


# ── 1. GradCAM3D ──────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3-D (video) CNNs.

    Registers forward and backward hooks on `target_layer` so that a single
    call to `generate_heatmap` runs the full forward + backward pass and
    returns a (T, H, W) attention volume.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_inp, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Returns:
            heatmaps  – (CLIP_LEN, 7, 7) float32 array, values in [0, 1]
            class_idx – predicted class index (0 = Real, 1 = Gen AI)
            probs     – (2,) softmax probabilities [p_real, p_genai]
        """
        self.model.eval()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Channel-wise gradient importance weights
        pooled = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        for c in range(self.activations.shape[1]):
            self.activations[:, c] *= pooled[c]

        heatmap = torch.relu(torch.mean(self.activations, dim=1).squeeze())

        # Upsample to full temporal resolution (T=16)
        if heatmap.ndim == 3:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            heatmap = nn.functional.interpolate(
                heatmap, size=(CLIP_LEN, 7, 7), mode="trilinear", align_corners=False
            )
            heatmap = heatmap.squeeze()

        heatmap = heatmap / (heatmap.max() + 1e-8)
        probs   = torch.softmax(logits, dim=1)[0]

        return (
            heatmap.cpu().detach().numpy(),
            class_idx,
            probs.cpu().detach().numpy(),
        )


# ── 2. Model helpers ──────────────────────────────────────────────────────────

def load_classifier(model_ckpt: str) -> nn.Module:
    """Load an R3D-18 checkpoint and move it to the global DEVICE."""
    model = torchvision.models.video.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
    return model.to(DEVICE).eval()


def _load_depth_clip(
    depth_paths: list[Path],
    clip_len: int = CLIP_LEN,
    size: int = FRAME_SIZE,
) -> tuple[torch.Tensor, list[Path]]:
    """
    Uniformly sample `clip_len` frames from `depth_paths`.
    Returns the normalised input tensor and the sampled Path list.
    """
    n      = len(depth_paths)
    stride = max(n / clip_len, 1)
    idxs   = [min(int(i * stride), n - 1) for i in range(clip_len)]
    sampled = [depth_paths[i] for i in idxs]

    imgs = []
    for p in sampled:
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32) / 255.0)

    clip   = np.transpose(np.stack(imgs, axis=0), (3, 0, 1, 2))  # (C, T, H, W)
    tensor = torch.from_numpy(clip).unsqueeze(0).to(DEVICE).float()
    tensor = (tensor - _MEAN) / _STD
    tensor.requires_grad_(True)
    return tensor, sampled


# ── 3. Geometric feature extraction ──────────────────────────────────────────

def compute_geometric_features(depth_paths: list[Path]) -> dict:
    """
    Derive three scalar cues from a sequence of colorized depth images.

    temporal_stability – mean absolute pixel change between consecutive frames.
                         Lower values → the depth field barely moves (a synthetic
                         "frozen geometry" signature).
    spatial_smoothness – mean Laplacian variance across frames.
                         Lower values → unusually flat/smooth depth surfaces.
    edge_density       – fraction of pixels flagged as edges by Canny.
                         Lower values → blurry or absent object boundaries.
    """
    grays = []
    for p in depth_paths:
        img  = cv2.imread(str(p), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        grays.append(gray)

    diffs = [np.mean(np.abs(grays[i + 1] - grays[i])) for i in range(len(grays) - 1)]
    temporal_stability = float(np.mean(diffs)) if diffs else 0.0

    lap_energies = [float(np.var(cv2.Laplacian(g, cv2.CV_32F))) for g in grays]
    spatial_smoothness = float(np.mean(lap_energies))

    edge_fracs = []
    for g in grays:
        edges = cv2.Canny((g * 255).astype(np.uint8), 50, 150)
        edge_fracs.append(float(np.mean(edges > 0)))
    edge_density = float(np.mean(edge_fracs))

    return {
        "temporal_stability": temporal_stability,
        "spatial_smoothness": spatial_smoothness,
        "edge_density":       edge_density,
    }


# ── 4. Explanation text builder ───────────────────────────────────────────────

def build_explanation_text(
    features: dict,
    pred_label: str,
    probs: np.ndarray,
) -> dict:
    """
    Convert numeric features into plain-language and technical bullet text.

    Returns:
        {"plain": list[str], "technical": list[str]}
    """
    ts = features["temporal_stability"]
    sm = features["spatial_smoothness"]
    ed = features["edge_density"]

    p_genai = float(probs[1]) * 100
    p_real  = float(probs[0]) * 100
    other_label = "real" if pred_label == "Gen AI" else "AI-generated"

    margin = abs(p_genai - p_real)
    if margin > 60:
        strength = "very strong"
    elif margin > 30:
        strength = "strong"
    elif margin > 15:
        strength = "moderate"
    else:
        strength = "weak"

    # ── Plain language ──────────────────────────────────────────────────────
    plain: list[str] = []

    plain.append(
        f"The model produced a {strength} {pred_label.lower()} signal "
        f"({p_genai:.1f}% AI vs {p_real:.1f}% real)."
    )

    if ts < _TS_LOW:
        plain.append(
            "The depth structure barely changes between frames — real-world scenes "
            "almost always contain some natural motion or camera shake."
        )
    elif ts > _TS_HIGH:
        plain.append(
            "Depth values shift noticeably across frames, which is consistent "
            "with natural motion in a real scene."
        )
    else:
        plain.append(
            "Frame-to-frame depth variation is in a moderate range — "
            "neither suspiciously static nor clearly dynamic."
        )

    if sm < _SM_LOW:
        plain.append(
            "The inferred geometry is unusually smooth and featureless — AI generators "
            "often produce over-regularised depth fields that lack real-world texture."
        )
    elif sm > _SM_HIGH:
        plain.append(
            "Rich surface detail is present in the depth map, which is "
            "characteristic of real-world scenes captured on camera."
        )
    else:
        plain.append(
            "Depth surface detail is at an intermediate level — "
            "not conclusive on its own."
        )

    if ed < _ED_LOW:
        plain.append(
            "Very few sharp object boundaries appear in the depth map. "
            "AI depth estimators sometimes fail to produce crisp edges at object silhouettes."
        )
    elif ed > _ED_HIGH:
        plain.append(
            "Well-defined object boundaries are visible throughout the depth map, "
            "indicating coherent real-world geometry."
        )
    else:
        plain.append(
            "Object boundary sharpness in the depth map is average and inconclusive."
        )

    plain.append(
        "Highlighted regions show where the model's attention was strongest — "
        "they are a gradient-based approximation, not a pixel-level authenticity mask."
    )

    # ── Technical detail ────────────────────────────────────────────────────
    technical: list[str] = [
        f"Prediction: {pred_label}  |  P(Gen AI) = {p_genai:.2f}%  |  P(Real) = {p_real:.2f}%",
        (
            f"Temporal stability (mean |Δdepth|): {ts:.5f}  "
            f"[< {_TS_LOW} very static  |  > {_TS_HIGH} dynamic]"
        ),
        (
            f"Spatial smoothness (Laplacian var):  {sm:.5f}  "
            f"[< {_SM_LOW} very smooth  |  > {_SM_HIGH} textured]"
        ),
        (
            f"Edge density (Canny fraction):       {ed:.4f}  "
            f"[< {_ED_LOW} sparse  |  > {_ED_HIGH} dense]"
        ),
        "Saliency method: Grad-CAM on layer4[1].conv2[0] of R3D-18 (depth-map clips).",
        "Depth maps: Depth Anything V2 — ViT-S encoder.",
    ]

    return {"plain": plain, "technical": technical}


# ── 5. Temporal GradCAM grid (PNG) ────────────────────────────────────────────

def generate_temporal_grid(
    heatmaps:    np.ndarray,
    depth_paths: list[Path],
    output_path: str,
    cell_size:   int = 112,
) -> str:
    """
    Save a 4 × 4 grid PNG with all 16 heatmap-overlaid depth frames.

    Per-cell subtitle shows the frame index and mean attention intensity,
    making it easy to spot *which* part of the clip drove the prediction.
    A shared colour legend runs along the bottom.
    """
    n_rows, n_cols = 4, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
    fig.suptitle(
        "Temporal GradCAM — All 16 Sampled Depth Frames",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for t in range(CLIP_LEN):
        row, col = divmod(t, n_cols)
        ax       = axes[row][col]

        depth_img = cv2.cvtColor(
            cv2.imread(str(depth_paths[t]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        depth_img = cv2.resize(depth_img, (cell_size, cell_size))

        hm       = cv2.resize(heatmaps[t], (cell_size, cell_size))
        hm_color = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
        )
        overlay = cv2.addWeighted(depth_img, 0.55, hm_color, 0.45, 0)

        ax.imshow(overlay)
        ax.set_title(f"Frame {t + 1}  |  attn {hm.mean():.3f}", fontsize=7.5)
        ax.axis("off")

    # Shared colour legend
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes, orientation="horizontal",
        fraction=0.025, pad=0.04,
        label="Attention intensity  (0 = none  →  1 = highest)",
    )

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return output_path


# ── 6. Temporal GradCAM GIF ───────────────────────────────────────────────────

def generate_temporal_gif(
    heatmaps:    np.ndarray,
    depth_paths: list[Path],
    output_path: str,
    fps:         int = 4,
    cell_size:   int = 224,
) -> str | None:
    """
    Save an animated GIF cycling through all 16 heatmap overlays.
    Returns None if imageio is not installed.
    """
    try:
        import imageio
    except ImportError:
        print(
            "[explainability_toolkit] imageio not found — skipping GIF.\n"
            "  Install with:  pip install imageio"
        )
        return None

    frames_out = []
    for t in range(CLIP_LEN):
        depth_img = cv2.cvtColor(
            cv2.imread(str(depth_paths[t]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        depth_img = cv2.resize(depth_img, (cell_size, cell_size))

        hm       = cv2.resize(heatmaps[t], (cell_size, cell_size))
        hm_color = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
        )
        overlay = cv2.addWeighted(depth_img, 0.55, hm_color, 0.45, 0)

        cv2.putText(
            overlay, f"Frame {t + 1}/16",
            (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
        frames_out.append(overlay)

    imageio.mimsave(output_path, frames_out, format="GIF", duration=1.0 / fps, loop=0)
    return output_path


# ── 7. Summary panel (Layer A) ────────────────────────────────────────────────

def _make_summary_panel(
    rgb_mid:     np.ndarray,
    depth_mid:   np.ndarray,
    heatmaps:    np.ndarray,
    mid_idx:     int,
    pred_label:  str,
    probs:       np.ndarray,
    explanation: dict,
    output_path: str,
) -> str:
    """
    Three-column figure (RGB | Depth | Heatmap overlay) with:
      - dual probability bar chart
      - plain-language bullet points
      - one-line technical summary
    """
    h, w = depth_mid.shape[:2]
    hm       = cv2.resize(heatmaps[mid_idx], (w, h))
    hm_color = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    )
    overlay = cv2.addWeighted(depth_mid, 0.6, hm_color, 0.4, 0)

    p_genai = float(probs[1]) * 100
    p_real  = float(probs[0]) * 100
    conf    = p_genai if pred_label == "Gen AI" else p_real

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[3.5, 2.5], hspace=0.40, wspace=0.15,
    )

    ax_rgb   = fig.add_subplot(gs[0, 0])
    ax_depth = fig.add_subplot(gs[0, 1])
    ax_heat  = fig.add_subplot(gs[0, 2])
    ax_text  = fig.add_subplot(gs[1, :])

    ax_rgb.imshow(rgb_mid)
    ax_rgb.set_title("Input Frame (RGB)", fontweight="bold", fontsize=11)
    ax_rgb.axis("off")

    ax_depth.imshow(depth_mid)
    ax_depth.set_title("Geometric Cue (Depth Map)", fontweight="bold", fontsize=11)
    ax_depth.axis("off")

    verdict_color = "#c0392b" if pred_label == "Gen AI" else "#27ae60"
    ax_heat.imshow(overlay)
    ax_heat.set_title(
        f"Model Attention\n{pred_label}  —  {conf:.1f}% confidence",
        fontweight="bold", fontsize=11, color=verdict_color,
    )
    ax_heat.axis("off")

    sm_cb = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm_cb.set_array([])
    plt.colorbar(sm_cb, ax=ax_heat, fraction=0.046, pad=0.04).set_label(
        "Attention intensity", fontsize=8
    )

    # ── Bullet text ────────────────────────────────────────────────────────
    plain_text = "\n".join(f"  • {b}" for b in explanation["plain"])
    # Compact technical line (skip first entry which duplicates the title)
    tech_line  = "  |  ".join(explanation["technical"][1:4])

    ax_text.axis("off")
    ax_text.text(
        0.01, 0.98, "What the model found:",
        transform=ax_text.transAxes, fontsize=11, fontweight="bold", va="top",
    )
    ax_text.text(
        0.01, 0.84, plain_text,
        transform=ax_text.transAxes, fontsize=9, va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", alpha=0.85),
    )
    ax_text.text(
        0.01, 0.06,
        f"Technical:  {tech_line}",
        transform=ax_text.transAxes, fontsize=7.5, va="bottom",
        color="#555555", style="italic",
    )

    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    return output_path


# ── 8. Master entry point ─────────────────────────────────────────────────────

def generate_layered_report(
    depth_dir:  str,
    rgb_dir:    str,
    output_dir: str,
    model_ckpt: str | None        = None,
    model:      nn.Module | None  = None,
    video_name: str               = "video",
    clip_len:   int               = CLIP_LEN,
) -> dict:
    """
    Full explainability pipeline.  Either `model_ckpt` or a pre-loaded `model`
    must be supplied (the latter avoids reloading weights on every call).

    Parameters
    ----------
    depth_dir  : Folder of colorized depth PNG frames (output of Depth Anything V2).
    rgb_dir    : Folder of original RGB frames (used for display only).
    output_dir : Destination for all saved artefacts.
    model_ckpt : Path to R3D-18 `.pt` checkpoint (ignored when `model` is supplied).
    model      : Pre-loaded, eval-mode R3D-18 (pass this from processor.py to avoid
                 loading the weights twice).
    video_name : Filename prefix for all outputs.
    clip_len   : Number of frames in the depth clip (must match training config).

    Returns
    -------
    dict with keys:
        video, prediction, prob_genai, prob_real, confidence,
        summary_panel_path, temporal_grid_path, temporal_gif_path,
        explanation  {plain: list[str], technical: list[str]},
        features     {temporal_stability, spatial_smoothness, edge_density}
    """
    if model is None and model_ckpt is None:
        raise ValueError("Provide either `model_ckpt` or a pre-loaded `model`.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Frame discovery ──────────────────────────────────────────────────────
    _img_exts = {".png", ".jpg", ".jpeg"}
    depth_frames = sorted(
        [p for p in Path(depth_dir).iterdir() if p.suffix.lower() in _img_exts]
    )
    rgb_frames = sorted(
        [p for p in Path(rgb_dir).iterdir() if p.suffix.lower() in _img_exts]
    )

    if not depth_frames:
        raise RuntimeError(f"No depth frames found in: {depth_dir}")

    # ── Model + GradCAM ──────────────────────────────────────────────────────
    if model is None:
        model = load_classifier(model_ckpt)

    target_layer = model.layer4[1].conv2[0]
    cam          = GradCAM3D(model, target_layer)

    input_tensor, sampled_depth = _load_depth_clip(depth_frames, clip_len)
    heatmaps, pred_idx, probs   = cam.generate_heatmap(input_tensor)

    pred_label = "Gen AI" if pred_idx == 1 else "Real"
    prob_genai = float(probs[1])
    prob_real  = float(probs[0])
    conf_pct   = float(probs[pred_idx]) * 100

    # ── Geometric features + explanation text ────────────────────────────────
    features    = compute_geometric_features(sampled_depth)
    explanation = build_explanation_text(features, pred_label, probs)

    # ── Middle frame visuals ─────────────────────────────────────────────────
    mid_idx  = clip_len // 2
    n_rgb    = len(rgb_frames)
    rgb_mid  = cv2.cvtColor(
        cv2.imread(str(rgb_frames[min(mid_idx, n_rgb - 1)])), cv2.COLOR_BGR2RGB
    )
    depth_mid = cv2.cvtColor(
        cv2.imread(str(sampled_depth[mid_idx])), cv2.COLOR_BGR2RGB
    )

    # ── Save artefacts ───────────────────────────────────────────────────────
    summary_path = str(output_dir / f"{video_name}_summary.png")
    grid_path    = str(output_dir / f"{video_name}_temporal_grid.png")
    gif_path     = str(output_dir / f"{video_name}_temporal.gif")

    _make_summary_panel(
        rgb_mid, depth_mid, heatmaps, mid_idx,
        pred_label, probs, explanation, summary_path,
    )
    generate_temporal_grid(heatmaps, sampled_depth, grid_path)
    gif_result = generate_temporal_gif(heatmaps, sampled_depth, gif_path)

    return {
        "video":               video_name,
        "prediction":          pred_label,
        "prob_genai":          round(prob_genai * 100, 2),
        "prob_real":           round(prob_real  * 100, 2),
        "confidence":          f"{conf_pct:.2f}%",
        "summary_panel_path":  summary_path,
        "temporal_grid_path":  grid_path,
        "temporal_gif_path":   gif_result,
        "explanation":         explanation,
        "features":            features,
    }


# ── 9. Standalone CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GenAI Video Detection — Explainability Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      required=True, help="Path to R3D-18 .pt checkpoint")
    parser.add_argument("--depth-dir",  required=True, help="Folder of colorized depth frames")
    parser.add_argument("--rgb-dir",    required=True, help="Folder of original RGB frames")
    parser.add_argument("--output-dir", default="explainability_output", help="Output folder")
    parser.add_argument("--name",       default="video", help="Prefix for output filenames")
    args = parser.parse_args()

    result = generate_layered_report(
        model_ckpt=args.model,
        depth_dir=args.depth_dir,
        rgb_dir=args.rgb_dir,
        output_dir=args.output_dir,
        video_name=args.name,
    )

    print("\n" + "=" * 50)
    print("  EXPLAINABILITY REPORT")
    print("=" * 50)
    print(f"  Prediction : {result['prediction']}")
    print(f"  P(Gen AI)  : {result['prob_genai']}%")
    print(f"  P(Real)    : {result['prob_real']}%")
    print()
    print("  Plain Language:")
    for b in result["explanation"]["plain"]:
        print(f"    • {b}")
    print()
    print("  Technical Details:")
    for t in result["explanation"]["technical"]:
        print(f"    {t}")
    print()
    print(f"  Summary panel  → {result['summary_panel_path']}")
    print(f"  Temporal grid  → {result['temporal_grid_path']}")
    print(f"  Temporal GIF   → {result['temporal_gif_path']}")
    print("=" * 50)
