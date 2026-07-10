"""
Comparison script between 3D CNN (R3D-18) and Video Transformer models
"""
import torch
import sys
sys.path.append('../3d_cnn')

from model import VideoTransformer

# Import 3D CNN model
import torchvision
import torch.nn as nn

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def analyze_3d_cnn():
    """Analyze R3D-18 model"""
    print("=" * 60)
    print("3D CNN (R3D-18) Analysis")
    print("=" * 60)
    
    model = torchvision.models.video.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    total, trainable = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size: {size_mb:.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 16, 112, 112)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print()
    
    return total, size_mb

def analyze_transformer():
    """Analyze Video Transformer model"""
    print("=" * 60)
    print("Video Transformer (ViViT) Analysis")
    print("=" * 60)
    
    model = VideoTransformer(
        img_size=112,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        num_classes=2,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    total, trainable = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size: {size_mb:.2f} MB")
    
    # Architecture details
    num_patches = model.patch_embed.num_patches
    print(f"\nArchitecture Details:")
    print(f"  - Number of patches: {num_patches}")
    print(f"  - Embedding dimension: 384")
    print(f"  - Number of transformer blocks: 8")
    print(f"  - Number of attention heads: 6")
    print(f"  - MLP ratio: 4.0")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 16, 112, 112)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nInput Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print()
    
    return total, size_mb

def compare_models():
    """Compare both models side by side"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    cnn_params, cnn_size = analyze_3d_cnn()
    trans_params, trans_size = analyze_transformer()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'3D CNN':<15} {'Transformer':<15}")
    print("-" * 60)
    print(f"{'Parameters':<30} {cnn_params:>14,} {trans_params:>14,}")
    print(f"{'Model Size (MB)':<30} {cnn_size:>14.2f} {trans_size:>14.2f}")
    print(f"{'Architecture Type':<30} {'Convolutional':<15} {'Attention':<15}")
    print(f"{'Receptive Field':<30} {'Local':<15} {'Global':<15}")
    print(f"{'Inductive Bias':<30} {'Strong':<15} {'Weak':<15}")
    print()
    
    # Parameter difference
    param_diff = ((trans_params - cnn_params) / cnn_params) * 100
    print(f"Parameter Difference: {param_diff:+.1f}%")
    
    if trans_params < cnn_params:
        print(f"✓ Transformer has {cnn_params - trans_params:,} fewer parameters")
    else:
        print(f"✗ Transformer has {trans_params - cnn_params:,} more parameters")
    
    print("\nKey Advantages:")
    print("\n3D CNN (R3D-18):")
    print("  ✓ Lower memory footprint")
    print("  ✓ Faster inference")
    print("  ✓ Better with limited data")
    print("  ✓ Strong spatial inductive bias")
    
    print("\nVideo Transformer:")
    print("  ✓ Global context modeling")
    print("  ✓ Better scalability with data")
    print("  ✓ Attention visualization")
    print("  ✓ State-of-the-art performance potential")
    print("  ✓ Flexible sequence handling")

if __name__ == "__main__":
    compare_models()
