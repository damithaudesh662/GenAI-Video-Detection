import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1,3,1,1).to(DEVICE)
# STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1,3,1,1).to(DEVICE)

# After (correct for (B,C,T,H,W)):
MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)


@torch.no_grad()
def predict_video_folder(model_ckpt, video_dir, clip_len=16, size=112):
    print(f"Predicting video folder: {video_dir}")
    model = torchvision.models.video.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
    model = model.to(DEVICE).eval()

    # load frames
    frames = sorted([p for p in Path(video_dir).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}])
    if len(frames) == 0:
        raise RuntimeError("No frames found")
    # sample indices
    n = len(frames)
    stride = max(n / clip_len, 1)
    idxs = [int(i * stride) for i in range(clip_len)]
    idxs = [min(i, n-1) for i in idxs]

    imgs = []
    for i in idxs:
        img = cv2.imread(str(frames[i]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32)/255.0)
    clip = np.stack(imgs, axis=0)  # (T,H,W,C)
    clip = np.transpose(clip, (3,0,1,2))  # (C,T,H,W)
    x = torch.from_numpy(clip).unsqueeze(0).to(DEVICE)  # (1,C,T,H,W)
    x = (x - MEAN) / STD
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    pred = int(prob.argmax().item())
    return pred, prob.cpu().numpy()



# pred, prob = predict_video_folder('best_r3d18_depthmaps_full.pt', 'output')
# print(f"output_depthmaps/input",pred, prob)