import os
import sys
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to sys.path to import existing modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Import existing functional components
from gen_ai_detector.process_video import process_video
from gen_ai_detector.depth_gen import make_depthmaps

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)

# --------------------------------------------------
# GradCAM3D Implementation (Refactored)
# --------------------------------------------------
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        class_loss = logits[0, class_idx]
        class_loss.backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        if len(heatmap.shape) == 3:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            heatmap = torch.nn.functional.interpolate(heatmap, size=(16, 7, 7), mode='trilinear', align_corners=False)
            heatmap = heatmap.squeeze()
        heatmap /= (torch.max(heatmap) + 1e-8)
        return heatmap.cpu().detach().numpy(), class_idx, torch.softmax(logits, dim=1)[0].cpu().detach().numpy()

# --------------------------------------------------
# Main Backend Functions
# --------------------------------------------------

def process_single_video(video_path, model_path, output_dir):
    """
    Processes a single video file:
    1. Extracts frames
    2. Generates depth maps
    3. Predicts Real vs AI
    4. Generates an explainable heatmap
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary directory for this video
    temp_dir = output_dir / f"temp_{video_path.stem}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        # 1. Process video (Extract frames)
        # Using 5s duration and 4fps as per earlier methodology
        process_video(video_path, t=5, fps=4, output_root=temp_dir)
        frame_dir = temp_dir / "frames"
        depth_dir = temp_dir / "depthmaps"

        # 2. Make depth maps
        make_depthmaps(frame_dir, depth_dir)

        # 3. Model Inference and Explainability
        model = torchvision.models.video.r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(DEVICE).eval()
        
        target_layer = model.layer4[1].conv2[0]
        cam = GradCAM3D(model, target_layer)

        # Load depth clip (16 frames)
        depth_frames = sorted([p for p in depth_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        if len(depth_frames) < 16:
            # Simple padding
            depth_frames += [depth_frames[-1]] * (16 - len(depth_frames))
        
        depth_imgs = []
        for i in range(16):
            img = cv2.imread(str(depth_frames[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
            depth_imgs.append(img.astype(np.float32)/255.0)
        
        clip = np.transpose(np.stack(depth_imgs, axis=0), (3,0,1,2))
        input_tensor = (torch.from_numpy(clip).unsqueeze(0).to(DEVICE) - MEAN) / STD
        input_tensor.requires_grad = True

        heatmaps, pred_idx, probs = cam.generate_heatmap(input_tensor)
        
        # 4. Save Final Heatmap Output
        result_label = "Gen AI" if pred_idx == 1 else "Real"
        confidence = probs[pred_idx] * 100
        
        # Prepare the feedback image (middle frame)
        mid_idx = 8
        orig_frame = cv2.imread(str(sorted(list(frame_dir.iterdir()))[mid_idx]))
        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        
        depth_frame = cv2.imread(str(depth_frames[mid_idx]))
        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
        
        h, w, _ = depth_frame.shape
        heatmap_img = cv2.resize(heatmaps[mid_idx], (w, h))
        heatmap_img = np.uint8(255 * heatmap_img)
        heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(depth_frame, 0.6, heatmap_colored, 0.4, 0)

        # Matplotlib for report generation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_frame); axes[0].set_title("Input RGB"); axes[0].axis('off')
        axes[1].imshow(depth_frame); axes[1].set_title("Geometric Cue"); axes[1].axis('off')
        axes[2].imshow(overlay); axes[2].set_title(f"Feedback ({result_label} {confidence:.1f}%)"); axes[2].axis('off')
        
        vis_path = output_dir / f"result_{video_path.stem}.png"
        plt.tight_layout()
        plt.savefig(vis_path)
        plt.close()

        return {
            "video": video_path.name,
            "prediction": result_label,
            "confidence": f"{confidence:.2f}%",
            "heatmap_path": str(vis_path)
        }

    except Exception as e:
        print(f"Error processing {video_path.name}: {e}")
        return {"video": video_path.name, "error": str(e)}
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def process_batch_videos(video_folder, model_path, output_root):
    """
    Processes all video files in a folder and saves separate heatmaps/reports.
    """
    video_folder = Path(video_folder)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = [p for p in video_folder.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}]
    print(f"Found {len(video_files)} videos in {video_folder}")

    batch_results = []
    for vid in video_files:
        print(f"--- Processing: {vid.name} ---")
        res = process_single_video(vid, model_path, output_root)
        batch_results.append(res)
    
    return batch_results

# Example Usage:
# if __name__ == "__main__":
#     model = "../best_r3d18_depthmaps_full.pt"
#     # Single
#     # print(process_single_video("test.mp4", model, "results"))
#     # Batch
#     # print(process_batch_videos("input_folder", model, "batch_results"))
