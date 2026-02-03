# Video Transformer Architecture Diagram

## Data Flow

```
Input Video (Depth Maps)
    ↓
[B, C=3, T=16, H=112, W=112]
    ↓
┌─────────────────────────────────────┐
│   3D Patch Embedding                │
│   - Temporal patches: 2 frames      │
│   - Spatial patches: 16×16 pixels   │
│   - Conv3D projection               │
└─────────────────────────────────────┘
    ↓
[B, num_patches=392, embed_dim=384]
    ↓
┌─────────────────────────────────────┐
│   Add [CLS] Token                   │
└─────────────────────────────────────┘
    ↓
[B, 393, 384]
    ↓
┌─────────────────────────────────────┐
│   Add Positional Embedding          │
│   (Learnable)                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Transformer Block 1               │
│   ┌───────────────────────────┐     │
│   │ Layer Norm                │     │
│   │ Multi-Head Self-Attention │     │
│   │ (6 heads)                 │     │
│   └───────────────────────────┘     │
│            ↓ (Residual)             │
│   ┌───────────────────────────┐     │
│   │ Layer Norm                │     │
│   │ MLP (GELU)                │     │
│   │ (384 → 1536 → 384)        │     │
│   └───────────────────────────┘     │
│            ↓ (Residual)             │
└─────────────────────────────────────┘
    ↓
    ... (Repeat 8 times)
    ↓
┌─────────────────────────────────────┐
│   Final Layer Norm                  │
└─────────────────────────────────────┘
    ↓
Extract [CLS] Token
    ↓
[B, 384]
    ↓
┌─────────────────────────────────────┐
│   Classification Head               │
│   Linear(384 → 2)                   │
└─────────────────────────────────────┘
    ↓
[B, 2] (Logits)
    ↓
Softmax
    ↓
[Real, Gen AI] Probabilities
```

## Patch Embedding Details

```
Video: [B, 3, 16, 112, 112]
       ↓
3D Conv (kernel=2×16×16, stride=2×16×16)
       ↓
[B, 384, 8, 7, 7]
       ↓
Flatten spatial-temporal dims
       ↓
[B, 384, 392]
       ↓
Transpose
       ↓
[B, 392, 384]

Where:
- 8 temporal patches (16 frames / 2)
- 7×7 spatial patches (112 / 16)
- Total: 8 × 7 × 7 = 392 patches
```

## Multi-Head Self-Attention

```
Input: [B, N=393, D=384]
       ↓
Linear Projection to Q, K, V
       ↓
[B, N, 3×D] → Reshape → [B, N, 3, H=6, D_h=64]
       ↓
Split into Q, K, V
       ↓
Q: [B, H=6, N, D_h=64]
K: [B, H=6, N, D_h=64]
V: [B, H=6, N, D_h=64]
       ↓
Attention = softmax(Q @ K^T / √D_h)
       ↓
[B, H, N, N] (Attention Map)
       ↓
Output = Attention @ V
       ↓
[B, H, N, D_h]
       ↓
Concatenate heads & Project
       ↓
[B, N, D=384]
```

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| img_size | 112 | Input image resolution |
| patch_size | 16 | Spatial patch size |
| temporal_patch_size | 2 | Temporal patch size |
| embed_dim | 384 | Embedding dimension |
| depth | 8 | Number of transformer blocks |
| num_heads | 6 | Attention heads per block |
| mlp_ratio | 4.0 | MLP hidden dim ratio |
| dropout | 0.1 | Dropout rate |

## Computational Complexity

### Self-Attention
- Time: O(N² × D) where N=393, D=384
- Space: O(N²) for attention matrix

### MLP
- Time: O(N × D × mlp_ratio × D)
- Space: O(N × D)

### Total per block
- ~2.4M FLOPs per sample
- ~8 blocks = ~19.2M FLOPs total

## Comparison with 3D CNN

```
3D CNN (R3D-18)              Video Transformer
─────────────────            ─────────────────
Conv3D layers                Patch Embedding
    ↓                            ↓
Residual Blocks              Transformer Blocks
    ↓                            ↓
Global Avg Pool              [CLS] Token
    ↓                            ↓
FC Layer                     Linear Head
    ↓                            ↓
Output                       Output

Local receptive field        Global receptive field
~33M parameters              ~15M parameters
Lower memory                 Higher memory
Faster inference             Slower inference
```
