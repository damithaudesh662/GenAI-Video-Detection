# Video Transformer for Depth Map Classification

This folder contains a **Video Vision Transformer (ViViT)** implementation for classifying videos as real or AI-generated based on depth maps.

## Architecture Overview

### Key Differences from 3D CNN (R3D-18)

| Component | 3D CNN (R3D-18) | Video Transformer (ViViT) |
|-----------|-----------------|---------------------------|
| **Architecture** | Convolutional layers with 3D kernels | Self-attention based transformer blocks |
| **Receptive Field** | Local (limited by kernel size) | Global (all patches attend to each other) |
| **Parameters** | ~33M parameters | ~15M parameters (configurable) |
| **Inductive Bias** | Strong spatial locality bias | Minimal inductive bias, learns from data |
| **Patch Processing** | Sliding window convolutions | Tokenized 3D patches |
| **Memory Usage** | Lower | Higher (due to attention) |

### Model Components

1. **3D Patch Embedding**
   - Converts video (B, C, T, H, W) into spatiotemporal patches
   - Temporal patch size: 2 frames
   - Spatial patch size: 16×16 pixels
   - Embedding dimension: 384

2. **Positional Encoding**
   - Learnable positional embeddings for each patch
   - Includes a special [CLS] token for classification

3. **Transformer Encoder**
   - 8 transformer blocks
   - 6 attention heads per block
   - Multi-head self-attention mechanism
   - Feed-forward MLP with GELU activation
   - Layer normalization (pre-norm architecture)

4. **Classification Head**
   - Linear layer mapping [CLS] token to 2 classes
   - Softmax for probability distribution

## Files

- **model.py**: Video Transformer architecture implementation
- **dataset.py**: Dataset loader for depth map videos
- **dataloader.py**: DataLoader utilities
- **main.py**: Training script
- **predict.py**: Inference script for trained models

## Usage

### Training

```python
python main.py
```

The script will:
1. Load depth map videos from `../3d_cnn/dataset/depthmaps/`
2. Train the transformer model for 20 epochs
3. Save the best model as `best_video_transformer_depthmaps.pt`

### Prediction

```python
from predict import predict_video_folder

pred, prob = predict_video_folder(
    'best_video_transformer_depthmaps.pt',
    'path/to/video/folder'
)
print(f"Prediction: {'Gen AI' if pred == 1 else 'Real'}")
print(f"Confidence: {prob[pred]:.2%}")
```

## Hyperparameters

- **Image Size**: 112×112
- **Clip Length**: 16 frames
- **Batch Size**: 2 (smaller due to memory requirements)
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing (T_max=20)
- **Dropout**: 0.1

## Advantages of Transformer Approach

1. **Global Context**: Self-attention allows the model to capture long-range dependencies across both spatial and temporal dimensions
2. **Flexibility**: Can handle variable-length sequences more naturally
3. **Interpretability**: Attention weights can be visualized to understand what the model focuses on
4. **Scalability**: Performance improves with more data and compute
5. **State-of-the-art**: Transformers have shown superior performance on many video understanding tasks

## Memory Considerations

Transformers require more GPU memory than CNNs due to the quadratic complexity of self-attention. If you encounter OOM errors:

- Reduce batch size (currently set to 2)
- Reduce embedding dimension (384 → 256)
- Reduce number of transformer blocks (8 → 6)
- Reduce number of attention heads (6 → 4)
- Increase patch size (16 → 32)

## Expected Performance

The transformer model should achieve comparable or better accuracy than the 3D CNN, especially with:
- Larger datasets
- Longer training
- Fine-tuning on domain-specific data

## Citation

This implementation is inspired by:
- **ViViT**: A Video Vision Transformer (Arnab et al., 2021)
- **Vision Transformer (ViT)**: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
