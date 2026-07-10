#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract Logits from Depth-Based Model (R3D18)
Outputs raw logits (before softmax) from the depth detection model for late fusion approaches.
Supports command-line configuration of video/depth folder and output paths.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pipeline constants — kept inline so this script has no external config dependency.
CLIP_LEN_S  = 5.0   # seconds trimmed per video
TARGET_FPS  = 4     # frames extracted per second
CLIP_FRAMES = 16    # temporal frames fed to the 3D-CNN
FRAME_SIZE  = 112   # spatial H×W expected by the model


def sample_indices(n, clip_frames=CLIP_FRAMES):
    """Uniform-stride sampling matching VideoDepthDataset._sample_indices."""
    if n >= clip_frames:
        stride = n / clip_frames
        idxs = [int(i * stride) for i in range(clip_frames)]
    else:
        idxs = list(range(n)) + [n - 1] * (clip_frames - n)
    return [min(max(i, 0), n - 1) for i in idxs]

# Ensure Unicode-safe console output on Windows (depth_gen prints '→')
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Default configuration
DEFAULT_NUM_CLASSES = 2
DEFAULT_FRAME_RATE  = TARGET_FPS
DEFAULT_DURATION    = CLIP_LEN_S


# -----------------------------------------------
# Depth Model (R3D18-based detector)
# -----------------------------------------------
class R3D18DepthDetector(nn.Module):
    """R3D18-based depth map video classifier (flat module tree, matches 3d_cnn checkpoints)."""

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        from torchvision.models.video import r3d_18

        # Same layout as Geometric_Clues/Depth-Anything-V2/3d_cnn/model.py (state_dict keys: stem.*, fc.*)
        try:
            self.net = r3d_18(weights=None)
        except TypeError:
            self.net = r3d_18(pretrained=False)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, channels, frames, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        return self.net(x)

    def get_features(self, x):
        """Get features before classification layer"""
        x = self.net.stem(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def _normalize_r3d_checkpoint_state_dict(state_dict):
    """Map checkpoints saved with nested wrappers or DataParallel to flat r3d_18 keys."""
    if state_dict is None:
        return state_dict
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        keys = list(state_dict.keys())
    if all(k.startswith("backbone.") for k in keys):
        state_dict = {k[len("backbone.") :]: v for k, v in state_dict.items()}
    elif all(k.startswith("net.") for k in keys):
        state_dict = {k[len("net.") :]: v for k, v in state_dict.items()}
    return state_dict


def _add_prefix_to_state_dict(state_dict, prefix: str):
    if not prefix:
        return state_dict
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


# -----------------------------------------------
# Depth Processing
# -----------------------------------------------
def process_video(video_path, frame_rate, duration, output_folder):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        frame_rate: Extract every Nth frame
        duration: Duration in seconds to sample
        output_folder: Where to save frames
    """
    try:
        from process_video import process_video as process_video_fn
        logger.info(f"Processing video: {video_path}")
        process_video_fn(video_path, frame_rate, duration, output_folder)
        logger.info(f"✓ Frames extracted to {output_folder}")
    except ImportError:
        logger.error("process_video module not found. Please ensure it's in the Python path.")
        raise


def make_depthmaps(frames_folder, output_folder):
    """
    Generate depth maps from frames.
    Loads gen_ai_detector/depth_gen.py by absolute path so that the root-level
    depth_gen.py (which uses a relative checkpoint path) is never picked up
    instead, regardless of sys.path ordering.
    """
    import importlib.util
    _depth_gen_path = Path(__file__).resolve().parent / "depth_gen.py"
    _spec = importlib.util.spec_from_file_location("depth_gen_local", _depth_gen_path)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    logger.info(f"Generating depth maps from {frames_folder}")
    _mod.make_depthmaps(frames_folder, output_folder)
    logger.info(f"✓ Depth maps generated to {output_folder}")


# -----------------------------------------------
# Logits Extractor for Depth Model
# -----------------------------------------------
class DepthLogitsExtractor:
    """Extract logits from depth-based 3D CNN model"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the depth logits extractor
        
        Args:
            model_path: Path to trained R3D18 model
            device: torch device to use
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading depth model from {model_path}...")

        self.model = R3D18DepthDetector(num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        else:
            sd = checkpoint

        sd = _normalize_r3d_checkpoint_state_dict(sd)

        # Our wrapper model has keys `net.*`, but most checkpoints are saved from a bare r3d_18
        # (keys like `stem.*`, `layer1.*`, `fc.*`). Add prefix if needed.
        model_keys = list(self.model.state_dict().keys())
        if model_keys and model_keys[0].startswith("net.") and not any(k.startswith("net.") for k in sd.keys()):
            sd = _add_prefix_to_state_dict(sd, "net.")

        self.model.load_state_dict(sd, strict=True)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Depth model loaded successfully on {self.device}")
    
    def extract_logits_from_depthmaps(self, depthmaps_folder):
        """
        Extract logits from depth maps in a folder
        
        Args:
            depthmaps_folder: Path to folder containing depth maps
            
        Returns:
            logits: numpy array of shape (2,) containing raw logits [logit_real, logit_genai]
        """
        depth_files = sorted(
            f
            for f in os.listdir(depthmaps_folder)
            if f.endswith((".npz", ".npy", ".pt", ".png", ".jpg", ".jpeg"))
        )

        if len(depth_files) == 0:
            raise ValueError(f"No depth maps found in {depthmaps_folder}")

        depth_frames = []

        for depth_file in depth_files:
            depth_path = os.path.join(depthmaps_folder, depth_file)

            if depth_file.endswith(".npz"):
                # Single-channel float16 [0,1] produced by batched depth_gen.
                # Replicate to 3 channels to match model input shape.
                raw = np.load(depth_path)['depth'].astype(np.float32)
                if raw.size == 0:
                    continue
                raw = cv2.resize(raw, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
                frame = np.stack([raw, raw, raw], axis=-1)  # H×W×3

            elif depth_file.endswith(".npy"):
                raw = np.load(depth_path).astype(np.float32)
                if raw.size == 0:
                    continue
                if raw.max() > 1.0:
                    raw = raw / 255.0
                raw = cv2.resize(raw, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
                frame = np.stack([raw, raw, raw], axis=-1) if raw.ndim == 2 else raw

            elif depth_file.endswith((".png", ".jpg", ".jpeg")):
                # 3-channel INFERNO colormap PNGs produced by automated.py.
                # Load as BGR → RGB and scale to [0,1], matching dataset._load_frame
                # in 3d_cnn/dataset.py so inference is consistent with training.
                img = cv2.imread(depth_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # H×W×3

            elif depth_file.endswith(".pt"):
                raw = torch.load(depth_path)
                if isinstance(raw, torch.Tensor):
                    raw = raw.cpu().numpy().astype(np.float32)
                if raw.size == 0:
                    continue
                if raw.max() > 1.0:
                    raw = raw / 255.0
                raw = cv2.resize(raw, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
                frame = np.stack([raw, raw, raw], axis=-1) if raw.ndim == 2 else raw

            else:
                continue

            depth_frames.append(frame)  # each frame is H×W×3 float32 in [0,1]

        if len(depth_frames) == 0:
            raise ValueError("Failed to load any depth maps")

        # Uniform-stride sampling to match VideoDepthDataset._sample_indices
        idxs = sample_indices(len(depth_frames))
        depth_frames = [depth_frames[i] for i in idxs]

        # (T, H, W, 3) → (3, T, H, W) → (1, 3, T, H, W)
        depth_stack = np.stack(depth_frames)                          # (T, H, W, 3)
        depth_tensor = torch.from_numpy(
            np.transpose(depth_stack, (3, 0, 1, 2))                   # (3, T, H, W)
        ).unsqueeze(0).to(self.device)                                 # (1, 3, T, H, W)

        # Kinetics-style normalization (matches 3d_cnn/model.py training)
        mean = torch.tensor(
            [0.43216, 0.394666, 0.37645], device=self.device, dtype=depth_tensor.dtype
        ).view(1, 3, 1, 1, 1)
        std = torch.tensor(
            [0.22803, 0.22145, 0.216989], device=self.device, dtype=depth_tensor.dtype
        ).view(1, 3, 1, 1, 1)
        depth_tensor = (depth_tensor - mean) / std

        with torch.no_grad():
            logits = self.model(depth_tensor)

        return logits.squeeze(0).cpu().numpy()


# -----------------------------------------------
# Results Saving and Visualization
# -----------------------------------------------
def save_logits_results(logits_data, output_folder, save_format='both'):
    """
    Save logits in multiple formats for flexibility
    
    Args:
        logits_data: Dictionary mapping video names to logits
        output_folder: Folder to save results
        save_format: Format to save ('npy', 'json', or 'both')
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if save_format in ['npy', 'both']:
        # Save as individual .npy files
        npy_folder = os.path.join(output_folder, 'npy')
        os.makedirs(npy_folder, exist_ok=True)
        
        for video_name, logits in logits_data.items():
            npy_path = os.path.join(npy_folder, f"{video_name}_logits.npy")
            np.save(npy_path, logits)
        
        logger.info(f"✓ Logits saved as .npy files in {npy_folder}")
    
    if save_format in ['json', 'both']:
        # Save as JSON for readability
        json_folder = os.path.join(output_folder, 'json')
        os.makedirs(json_folder, exist_ok=True)
        
        for video_name, logits in logits_data.items():
            json_path = os.path.join(json_folder, f"{video_name}_logits.json")
            json_data = {
                'video_name': video_name,
                'logits': logits.tolist(),
                'logit_real': float(logits[0]),
                'logit_genai': float(logits[1]),
                'source': 'depth_model'
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        logger.info(f"✓ Logits saved as .json files in {json_folder}")
    
    # Save summary file
    summary_path = os.path.join(output_folder, 'depth_logits_summary.json')
    summary_data = {
        'num_videos': len(logits_data),
        'model_source': 'depth_detector_r3d18',
        'videos': {}
    }
    
    for video_name, logits in logits_data.items():
        summary_data['videos'][video_name] = {
            'logit_real': float(logits[0]),
            'logit_genai': float(logits[1]),
            'diff': float(logits[1] - logits[0]),  # Positive = leans toward GenAI
        }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"✓ Summary saved to {summary_path}")


def find_video_files_recursive(video_folder: str):
    """All videos under folder (recursive), including real/ and fake/ subfolders."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")
    root = Path(video_folder)
    found = []
    for ext in video_extensions:
        found.extend(root.rglob(f"*{ext}"))
        found.extend(root.rglob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in found if p.is_file()})


def visualize_logits(logits_data, output_folder):
    """
    Visualize logits distribution across videos
    
    Args:
        logits_data: Dictionary mapping video names to logits
        output_folder: Folder to save visualizations
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract logits for plotting
    video_names = list(logits_data.keys())
    logits_real = [logits_data[name][0] for name in video_names]
    logits_genai = [logits_data[name][1] for name in video_names]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Logits comparison
    x_pos = np.arange(len(video_names))
    axes[0].bar(x_pos - 0.2, logits_real, 0.4, label='Real', alpha=0.8, color='blue')
    axes[0].bar(x_pos + 0.2, logits_genai, 0.4, label='GenAI', alpha=0.8, color='red')
    axes[0].set_xlabel('Video')
    axes[0].set_ylabel('Logit Value')
    axes[0].set_title('Depth Model (R3D18) - Logits Comparison Across Videos')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(video_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Logit differences (GenAI - Real)
    logit_diff = [logits_genai[i] - logits_real[i] for i in range(len(logits_real))]
    colors = ['red' if x > 0 else 'blue' for x in logit_diff]
    axes[1].bar(x_pos, logit_diff, color=colors, alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Video')
    axes[1].set_ylabel('Logit Difference (GenAI - Real)')
    axes[1].set_title('Depth Model - Logit Difference Distribution\n(Positive = Leans Toward GenAI, Negative = Leans Toward Real)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(video_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'depth_logits_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Depth logits visualization saved to {plot_path}")


# -----------------------------------------------
# Main Processing
# -----------------------------------------------
def find_depth_folders(base_dir):
    """
    Find all depth map folders in directory
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of (depth_folder, folder_name) tuples
    """
    depth_folders = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if directory contains depth maps
            files = os.listdir(item_path)
            if any(f.endswith(('.npz', '.npy', '.pt', '.png', '.jpg')) for f in files):
                depth_folders.append((item_path, item))
    
    return sorted(depth_folders)


def main():
    parser = argparse.ArgumentParser(
        description='Extract logits from depth-based detector (R3D18) for late fusion'
    )
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained R3D18 depth model (.pt file)')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Path to folder where logits will be saved')
    
    # Input options - either depth maps or videos
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--depth-folder', type=str,
                            help='Path to folder containing depth maps')
    input_group.add_argument('--video-folder', type=str,
                            help='Path to folder containing videos (will generate depth maps)')
    input_group.add_argument('--depthmaps-base-dir', type=str,
                            help='Base directory containing multiple depth map folders')
    
    # Optional arguments for video processing
    parser.add_argument('--frame-rate', type=int, default=DEFAULT_FRAME_RATE,
                       help=f'Frame rate for extraction (default: {DEFAULT_FRAME_RATE})')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION,
                       help=f'Duration in seconds to sample (default: {DEFAULT_DURATION})')
    
    # Processing options
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (auto: cuda if available, else cpu)')
    parser.add_argument('--save-format', type=str, default='both',
                       choices=['npy', 'json', 'both'],
                       help='Format to save logits (default: both)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots of logits')
    parser.add_argument('--skip-errors', action='store_true',
                       help='Skip folders that cause errors instead of stopping')
    parser.add_argument('--keep-frames', action='store_true',
                       help='Keep extracted frames (only when --video-folder is used)')
    parser.add_argument('--keep-depthmaps', action='store_true',
                       help='Keep generated depth maps (only when --video-folder is used)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Print configuration
    logger.info("=" * 70)
    logger.info("DEPTH MODEL LOGITS EXTRACTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output Folder: {args.output_folder}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Initialize extractor
        logger.info("Step 1: Initializing model...")
        logger.info("-" * 70)
        extractor = DepthLogitsExtractor(args.model_path, device=device)
        logger.info("")
        
        # Determine input mode and get depth folders
        if args.depth_folder:
            # Direct depth maps folder
            if not os.path.exists(args.depth_folder):
                logger.error(f"Depth folder not found: {args.depth_folder}")
                sys.exit(1)
            
            depth_folders = [(args.depth_folder, os.path.basename(args.depth_folder))]
        
        elif args.video_folder:
            # Process videos to depth maps
            if not os.path.exists(args.video_folder):
                logger.error(f"Video folder not found: {args.video_folder}")
                sys.exit(1)
            
            logger.info("Step 2: Processing videos to extract frames and depth maps...")
            logger.info("-" * 70)
            
            # Create temporary directories
            temp_frames = os.path.join(args.output_folder, '.temp', 'frames')
            temp_depthmaps = os.path.join(args.output_folder, '.temp', 'depthmaps')
            
            os.makedirs(temp_frames, exist_ok=True)
            os.makedirs(temp_depthmaps, exist_ok=True)
            
            video_paths = find_video_files_recursive(args.video_folder)

            if len(video_paths) == 0:
                logger.error(f"No video files found in {args.video_folder}")
                sys.exit(1)

            logger.info(f"✓ Found {len(video_paths)} video(s)")
            logger.info("")

            # Process each video
            depth_folders = []
            for video_path in tqdm(video_paths, desc="Processing videos"):
                video_path = str(video_path)
                video_name = Path(video_path).stem
                
                video_frames = os.path.join(temp_frames, video_name)
                video_depthmaps = os.path.join(temp_depthmaps, video_name)
                
                try:
                    process_video(video_path, TARGET_FPS, CLIP_LEN_S, video_frames)
                    # `process_video` writes frames into a nested `frames/` directory.
                    frames_dir = os.path.join(video_frames, "frames")
                    if os.path.isdir(frames_dir):
                        make_depthmaps(frames_dir, video_depthmaps)
                    else:
                        make_depthmaps(video_frames, video_depthmaps)
                    depth_folders.append((video_depthmaps, video_name))
                except Exception as e:
                    logger.warning(f"Failed to process {video_name}: {e}")
            
            logger.info("")
        
        elif args.depthmaps_base_dir:
            # Multiple depth map folders
            if not os.path.exists(args.depthmaps_base_dir):
                logger.error(f"Base directory not found: {args.depthmaps_base_dir}")
                sys.exit(1)
            
            logger.info("Step 2: Finding depth map folders...")
            logger.info("-" * 70)
            depth_folders = find_depth_folders(args.depthmaps_base_dir)
            
            if len(depth_folders) == 0:
                logger.error(f"No depth map folders found in {args.depthmaps_base_dir}")
                sys.exit(1)
            
            logger.info(f"✓ Found {len(depth_folders)} depth map folder(s)")
            logger.info("")
        
        # Extract logits
        logger.info("Step 3: Extracting logits...")
        logger.info("-" * 70)
        
        logits_data = {}
        skipped = 0
        
        for depth_folder, folder_name in tqdm(depth_folders, desc="Processing depth folders"):
            try:
                logits = extractor.extract_logits_from_depthmaps(depth_folder)
                logits_data[folder_name] = logits
            except Exception as e:
                error_msg = f"Failed to extract logits for {folder_name}: {e}"
                if args.skip_errors:
                    logger.warning(error_msg)
                    skipped += 1
                else:
                    logger.error(error_msg)
                    raise
        
        logger.info(f"✓ Extracted logits for {len(logits_data)} video(s)")
        if skipped > 0:
            logger.info(f"  (Skipped {skipped} folders due to errors)")
        logger.info("")
        
        if len(logits_data) == 0:
            logger.error("No logits were extracted successfully")
            sys.exit(1)
        
        # Save results
        logger.info("Step 4: Saving results...")
        logger.info("-" * 70)
        save_logits_results(logits_data, args.output_folder, args.save_format)
        logger.info("")
        
        # Visualize (optional)
        if args.visualize:
            logger.info("Step 5: Generating visualizations...")
            logger.info("-" * 70)
            visualize_logits(logits_data, args.output_folder)
            logger.info("")
        
        # Cleanup (if not keeping temporary files)
        if args.video_folder:
            temp_base = os.path.join(args.output_folder, '.temp')
            if not args.keep_frames or not args.keep_depthmaps:
                logger.info("Step 6: Cleaning up temporary files...")
                logger.info("-" * 70)
                
                if not args.keep_frames:
                    frames_dir = os.path.join(temp_base, 'frames')
                    if os.path.exists(frames_dir):
                        import shutil
                        shutil.rmtree(frames_dir)
                        logger.info("✓ Cleaned up frames")
                
                if not args.keep_depthmaps:
                    depthmaps_dir = os.path.join(temp_base, 'depthmaps')
                    if os.path.exists(depthmaps_dir):
                        import shutil
                        shutil.rmtree(depthmaps_dir)
                        logger.info("✓ Cleaned up depth maps")
                
                logger.info("")
        
        # Print summary
        logger.info("=" * 70)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Videos processed: {len(logits_data)}")
        logger.info(f"Output folder: {args.output_folder}")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - Logits (format: {args.save_format})")
        logger.info(f"  - Summary: depth_logits_summary.json")
        if args.visualize:
            logger.info(f"  - Visualization: depth_logits_visualization.png")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
