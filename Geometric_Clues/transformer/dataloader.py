from torch.utils.data import DataLoader
from dataset import VideoDepthDataset

def build_loaders(root, clip_len=16, size=112, batch_size=4, num_workers=2):
    train_ds = VideoDepthDataset(root, split='train', clip_len=clip_len, size=size)
    print(f"Training samples: {len(train_ds)}")
    val_ds = VideoDepthDataset(root, split='val', clip_len=clip_len, size=size)
    print(f"Validation samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    print(f"Built train loader with {len(train_loader)} batches")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f"Built val loader with {len(val_loader)} batches")
    return train_loader, val_loader
