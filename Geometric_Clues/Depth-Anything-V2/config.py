"""
Shared pipeline configuration
──────────────────────────────
Single source of truth for all values that must be identical across
dataset generation (automated.py), training (3d_cnn/), and inference
(backend/processor.py, gen_ai_detector/extract_logits_depth.py).

Import example:
    from config import CLIP_LEN_S, TARGET_FPS, CLIP_FRAMES, FRAME_SIZE, sample_indices
"""

# ── Video preparation ─────────────────────────────────────────────────────────
CLIP_LEN_S  = 5.0   # seconds to trim each source video to
TARGET_FPS  = 4     # frames extracted per second  →  5 s × 4 fps = 20 raw frames

# ── Model input ───────────────────────────────────────────────────────────────
CLIP_FRAMES = 16    # temporal depth fed to the 3D-CNN  (T dimension)
FRAME_SIZE  = 112   # spatial H × W for model input


def sample_indices(n, clip_frames=CLIP_FRAMES):
    """
    Uniform-stride frame sampling — identical to VideoDepthDataset._sample_indices.

    Selects `clip_frames` indices spread evenly over `n` available frames.
    When n < clip_frames the last frame index is repeated to pad to clip_frames.

    Args:
        n:           total number of available frames
        clip_frames: how many indices to return (default: CLIP_FRAMES)

    Returns:
        list of int indices, length == clip_frames
    """
    if n >= clip_frames:
        stride = n / clip_frames
        idxs = [int(i * stride) for i in range(clip_frames)]
    else:
        idxs = list(range(n)) + [n - 1] * (clip_frames - n)
    return [min(max(i, 0), n - 1) for i in idxs]
