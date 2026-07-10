from dataloader import build_loaders
from model import model, train
from torch.utils.data import DataLoader
from dataset import VideoDepthDataset
from predict import predict_video_folder
import torch
from pathlib import Path


if __name__ == "__main__":
    # Guard all multiprocessing-related code
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Build absolute path to dataset
    root = script_dir / ".." / "dataset" / "depthmaps"
    root = str(root.resolve())  # Convert to absolute path string
    print(f"Dataset path: {root}")
    train_loader, val_loader = build_loaders(root, num_workers=2, batch_size=2)  # Smaller batch for transformer
    xb, yb = next(iter(train_loader))
    print(xb.shape, yb.shape)
    print("Data loaders built.")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the model
    train(model, train_loader, val_loader, epochs=20)
    
    # Example prediction
    # pred, prob = predict_video_folder('best_video_transformer_depthmaps.pt', '../3d_cnn/dataset/depthmaps/gen_ai/1')
    # print(pred, prob)
