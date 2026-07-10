from dataloader import build_loaders

from model import model, train
from torch.utils.data import DataLoader
from dataset import VideoDepthDataset
from predict import predict_video_folder
import torch



if __name__ == "__main__":
    # Guard all multiprocessing-related code
    root = "dataset/depthmaps"
    train_loader, val_loader = build_loaders(root, num_workers=2)
    xb, yb = next(iter(train_loader))
    print(xb.shape, yb.shape)
    print("Data loaders built.")
    train(model, train_loader, val_loader, epochs=10)
    
    # Example
    # pred, prob = predict_video_folder('best_r3d18_depthmaps.pt', 'dataset/depthmaps/gen_ai/1')
    # print(pred, prob)
    
    