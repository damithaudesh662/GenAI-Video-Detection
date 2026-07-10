import torch
import torch.nn as nn
import torch.optim as optim
import math

NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device:{DEVICE}")


class PatchEmbedding3D(nn.Module):
    """
    Converts video (B, C, T, H, W) into patches and embeds them.
    """
    def __init__(self, img_size=112, patch_size=16, temporal_patch_size=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_temporal_patches = 16 // temporal_patch_size  # assuming 16 frames
        self.num_patches = self.num_patches_per_frame * self.num_temporal_patches
        
        # 3D convolution to extract patches
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size)
        )
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2)  # (B, embed_dim, T'*H'*W')
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoTransformer(nn.Module):
    """
    Video Vision Transformer (ViViT-style) for video classification
    """
    def __init__(
        self, 
        img_size=112, 
        patch_size=16, 
        temporal_patch_size=2,
        in_channels=3, 
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(
            img_size, patch_size, temporal_patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Use cls token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits


# Create model instance
model = VideoTransformer(
    img_size=112,
    patch_size=16,
    temporal_patch_size=2,
    in_channels=3,
    num_classes=NUM_CLASSES,
    embed_dim=384,  # Smaller for efficiency
    depth=8,  # Fewer layers than ViT-Base
    num_heads=6,
    mlp_ratio=4.0,
    dropout=0.1
).to(DEVICE)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Normalization constants (same as 3D CNN)
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
            torch.save(model.state_dict(), 'best_video_transformer_depthmaps.pt')
            print(f"Saved new best (acc={best_acc:.4f})")
