import cv2
import torch
import numpy as np
import os

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vitl', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Input/output folders
input_folder = "frames_output"
output_folder = "gen ai"
os.makedirs(output_folder, exist_ok=True)


# Supported image extensions
extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Process all images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(extensions):
        img_path = os.path.join(input_folder, filename)
        raw_img = cv2.imread(img_path)

        if raw_img is None:
            print(f"⚠️ Could not read {filename}, skipping...")
            continue

        # Run depth inference
        depth = model.infer_image(raw_img)

        # Normalize to 0–255 for visualization
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth_vis = depth_vis.astype(np.uint8)

        # Apply colormap
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # Save outputs
        base_name = os.path.splitext(filename)[0]
        # cv2.imwrite(os.path.join(output_folder, f"{base_name}_depth_gray.png"), depth_vis)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_depth_color.png"), depth_color)

        print(f"✅ Processed {filename}")

print("🎉 All images processed and saved to 'processed' folder.")