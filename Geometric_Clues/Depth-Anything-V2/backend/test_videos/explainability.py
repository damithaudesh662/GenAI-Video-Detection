import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        self.model.eval()
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = logits[0, class_idx]
        class_loss.backward()
        
        # Global Average Pooling of gradients
        # gradients shape: (B, C, T, H, W) -> (B, 512, 2, 7, 7)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        
        # Weight activations by gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :, :] *= pooled_gradients[i]
            
        # Create heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze() # (T_small, H_small, W_small)
        heatmap = torch.relu(heatmap)
        
        # Upsample heatmap to match original temporal resolution (T=16)
        # heatmap shape is (T_small, H_small, W_small), we need (T_orig, H_small, W_small)
        if len(heatmap.shape) == 3:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0) # (1, 1, T_small, H_s, W_s)
            heatmap = torch.nn.functional.interpolate(heatmap, size=(16, 7, 7), mode='trilinear', align_corners=False)
            heatmap = heatmap.squeeze() # (16, 7, 7)

        heatmap /= (torch.max(heatmap) + 1e-8)
        
        return heatmap.cpu().detach().numpy(), class_idx, torch.softmax(logits, dim=1)[0].cpu().detach().numpy()

def generate_visual_feedback(model_ckpt, rgb_dir, depth_dir, output_path="feedback_report.png"):
    """
    Generates a Triple-Panel (RGB - Depth - Heatmap) visualization
    """
    # 1. Load Model
    model = torchvision.models.video.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    # Target the last convolutional layer
    target_layer = model.layer4[1].conv2[0] 
    cam = GradCAM3D(model, target_layer)

    # 2. Prepare Data
    rgb_frames = sorted([p for p in Path(rgb_dir).iterdir() if p.suffix.lower() in {'.jpg', '.png'}])
    depth_frames = sorted([p for p in Path(depth_dir).iterdir() if p.suffix.lower() in {'.jpg', '.png'}])
    
    # Sample center clip (16 frames)
    clip_len = 16
    start_idx = max(0, len(depth_frames)//2 - clip_len//2)
    indices = range(start_idx, start_idx + clip_len)
    
    # Load Depth Clip for Prediction
    depth_imgs = []
    for i in indices:
        img = cv2.imread(str(depth_frames[i]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        depth_imgs.append(img.astype(np.float32)/255.0)
    
    depth_clip = np.stack(depth_imgs, axis=0)  # (T,H,W,C)
    depth_clip = np.transpose(depth_clip, (3,0,1,2))  # (C,T,H,W)
    input_tensor = torch.from_numpy(depth_clip).unsqueeze(0).to(DEVICE)
    input_tensor = (input_tensor - MEAN) / STD
    input_tensor.requires_grad = True

    # 3. Process explainability
    heatmaps, pred_idx, probs = cam.generate_heatmap(input_tensor)
    
    # 4. Create Visualization (Picking the middle frame of the clip)
    mid_idx = 8
    target_rgb = cv2.imread(str(rgb_frames[indices[mid_idx]]))
    target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)
    
    target_depth = cv2.imread(str(depth_frames[indices[mid_idx]]))
    target_depth = cv2.cvtColor(target_depth, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image size
    h, w, _ = target_depth.shape
    heatmap_img = cv2.resize(heatmaps[mid_idx], (w, h))
    heatmap_img = np.uint8(255 * heatmap_img)
    heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(target_depth, 0.6, heatmap_colored, 0.4, 0)

    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    class_name = "Gen AI" if pred_idx == 1 else "Real"
    confidence = probs[pred_idx] * 100

    axes[0].imshow(target_rgb)
    axes[0].set_title("Input Video (RGB)")
    axes[0].axis('off')

    axes[1].imshow(target_depth)
    axes[1].set_title("Geometric Cue (Depth Map)")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"Model Feedback (Heatmap)\nPrediction: {class_name} ({confidence:.2f}%)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Explainability report saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    generate_visual_feedback(
        model_ckpt='best_r3d18_depthmaps_full.pt',
        rgb_dir='dataset/real/4',           # Path to your RGB frames
        depth_dir='dataset/depthmaps/real/4',      # Path to your Depth maps
        output_path='detection_feedback.png'
    )
