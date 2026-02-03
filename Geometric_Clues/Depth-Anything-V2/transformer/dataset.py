import os
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDepthDataset(Dataset):
    def __init__(self, root, split='train', clip_len=16, size=112, seed=42, train_ratio=0.8):
        self.root = Path(root)  # e.g., dataset/depthmaps
        self.clip_len = clip_len
        self.size = size
        random.seed(seed)

        # Collect video directories and labels
        items = []
        for label_name, label in [('real', 0), ('gen_ai', 1)]:
            class_dir = self.root / label_name
            if not class_dir.exists():
                continue
            for vid_dir in class_dir.iterdir():
                if vid_dir.is_dir():
                    frames = sorted([p for p in vid_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
                    if len(frames) == 0:
                        continue
                    items.append({
                        'frames': frames,
                        'label': label,
                        'vid_dir': vid_dir,
                    })
        # Split train/val by video
        random.shuffle(items)
        n_train = int(len(items) * train_ratio)
        if split == 'train':
            self.items = items[:n_train]
        else:
            self.items = items[n_train:]

    def __len__(self):
        return len(self.items)

    def _sample_indices(self, n):
        # uniform sample clip_len indices over n frames
        if n >= self.clip_len:
            # choose stride to cover the whole video
            stride = n / self.clip_len
            idxs = [int(i * stride) for i in range(self.clip_len)]
        else:
            # pad by repeating last index
            idxs = list(range(n)) + [n-1] * (self.clip_len - n)
        # clamp
        idxs = [min(max(i, 0), n-1) for i in idxs]
        return idxs

    def _load_frame(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            # fallback black
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # convert BGR->RGB and scale to [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = item['frames']
        label = item['label']
        n = len(frames)
        idxs = self._sample_indices(n)
        imgs = [self._load_frame(frames[i]) for i in idxs]
        # shape (T, H, W, C) -> (C, T, H, W)
        clip = np.stack(imgs, axis=0)  # (T,H,W,C)
        clip = np.transpose(clip, (3,0,1,2))  # (C,T,H,W)
        clip = torch.from_numpy(clip)  # float32
        label = torch.tensor(label, dtype=torch.long)
        return clip, label
