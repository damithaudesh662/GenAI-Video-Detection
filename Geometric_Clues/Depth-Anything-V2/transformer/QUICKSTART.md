# Quick Start Guide - Video Transformer

## 🚀 Getting Started

### 1. Installation

```bash
cd transformer
pip install -r requirements.txt
```

### 2. Verify Setup

Run the model comparison to ensure everything is working:

```bash
python compare_models.py
```

This will show you the architecture details and parameter counts for both models.

### 3. Training

```bash
python main.py
```

**Note**: Make sure the dataset is available at `../3d_cnn/dataset/depthmaps/`

Expected output:
```
Training samples: 1479
Validation samples: 370
Built train loader with 739 batches
Built val loader with 185 batches
torch.Size([2, 3, 16, 112, 112]) torch.Size([2])
Data loaders built.
Total parameters: 15,234,562
Trainable parameters: 15,234,562
Starting training...
Epoch 1: loss=0.6931 val_acc=0.5000
...
```

### 4. Prediction

After training, use the saved model for inference:

```python
from predict import predict_video_folder

# Predict on a video folder
pred, prob = predict_video_folder(
    'best_video_transformer_depthmaps.pt',
    '../3d_cnn/dataset/depthmaps/gen_ai/1'
)

print(f"Prediction: {'Gen AI' if pred == 1 else 'Real'}")
print(f"Confidence: {prob[pred]:.2%}")
```

## 📊 Model Architecture

The Video Transformer uses a **ViViT (Video Vision Transformer)** architecture:

1. **Input**: Video clips of 16 frames (112×112 resolution)
2. **Patch Embedding**: Converts video into 392 spatiotemporal patches
3. **Transformer Encoder**: 8 layers with 6-head self-attention
4. **Classification**: Uses [CLS] token for final prediction

See `ARCHITECTURE.md` for detailed architecture diagrams.

## 🔧 Hyperparameter Tuning

### Reduce Memory Usage

If you encounter OOM (Out of Memory) errors:

```python
# In model.py, reduce these values:
embed_dim=256      # Instead of 384
depth=6            # Instead of 8
num_heads=4        # Instead of 6

# In main.py:
batch_size=1       # Instead of 2
```

### Improve Performance

For better accuracy (requires more compute):

```python
# In model.py:
embed_dim=512      # Larger embedding
depth=12           # More layers
num_heads=8        # More attention heads

# In main.py:
epochs=50          # Train longer
lr=5e-5            # Lower learning rate
```

## 📈 Expected Results

| Metric | Expected Value |
|--------|---------------|
| Training Time | ~2-3 hours (GPU) |
| Validation Accuracy | 85-95% |
| Model Size | ~60 MB |
| Inference Time | ~50-100ms per video |

## 🆚 Comparison with 3D CNN

| Aspect | 3D CNN | Transformer |
|--------|--------|-------------|
| Parameters | 33M | 15M |
| Training Speed | Faster | Slower |
| Inference Speed | Faster | Slower |
| Accuracy (large data) | Good | Better |
| Accuracy (small data) | Better | Good |
| Memory Usage | Lower | Higher |

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch_size to 1 or reduce model size

### Issue: Low Accuracy
**Solution**: 
- Train for more epochs
- Check data quality
- Try data augmentation
- Adjust learning rate

### Issue: Slow Training
**Solution**:
- Reduce num_workers in dataloader
- Use mixed precision training (add to model.py):
  ```python
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()
  ```

## 📚 Additional Resources

- **README.md**: Comprehensive documentation
- **ARCHITECTURE.md**: Detailed architecture diagrams
- **compare_models.py**: Side-by-side model comparison

## 💡 Tips

1. **Start Small**: Begin with default hyperparameters
2. **Monitor Training**: Watch for overfitting (val_acc plateaus)
3. **Save Checkpoints**: Model saves automatically when validation improves
4. **Visualize Attention**: Add attention visualization for interpretability
5. **Experiment**: Try different patch sizes, depths, and embedding dimensions

## 🎯 Next Steps

1. ✅ Train the baseline model
2. ✅ Evaluate on test set
3. ⬜ Implement data augmentation
4. ⬜ Add attention visualization
5. ⬜ Try ensemble with 3D CNN
6. ⬜ Fine-tune on domain-specific data

---

**Happy Training! 🚀**
