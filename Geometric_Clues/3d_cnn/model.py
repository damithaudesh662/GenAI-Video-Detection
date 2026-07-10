import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model: R3D-18
model = torchvision.models.video.r3d_18(weights=None)  # start from scratch; set weights=... to use pretrained
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Normalization constants (Kinetics-style)
# MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1,3,1,1).to(DEVICE)
# STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1,3,1,1).to(DEVICE)

# def normalize_clip(x):
#     # x: (B,C,T,H,W) in [0,1]
#     return (x - MEAN) / STD

# After (correct for (B,C,T,H,W)):
MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)

def normalize_clip(x):
    # x: (B, C, T, H, W) in [0,1]
    return (x - MEAN) / STD

@torch.no_grad()
def evaluate(model, loader):
    print("Evaluating...")
    model.eval()
    total, correct = 0, 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        xb = normalize_clip(xb)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
        print(f"Processed {total} samples...", end='\r')
    print("Evaluation done.")
    return correct / max(total, 1)

# Training
def train(model, train_loader, val_loader, epochs=20):
    best_acc = 0.0
    print("Starting training...")
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            xb = normalize_clip(xb)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
        scheduler.step()

        val_acc = evaluate(model, val_loader)
        epoch_loss = running_loss / max(len(train_loader.dataset), 1)
        print(f"Epoch {epoch}: loss={epoch_loss:.4f} val_acc={val_acc:.4f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_r3d18_depthmaps_full.pt')
            print(f"Saved new best (acc={best_acc:.4f})")


# train(model, train_loader, val_loader, epochs=10)
