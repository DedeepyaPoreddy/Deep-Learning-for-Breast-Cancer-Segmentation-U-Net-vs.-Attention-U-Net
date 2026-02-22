# Deep Learning for Breast Cancer Segmentation: U-Net vs. Attention U-Net

This repository compares two deep-learning segmentation models **U-Net** and **Attention U-Net** for breast cancer lesion mask segmentation using TensorFlow/Keras.

## Project overview
Goal: segment breast cancer regions (binary masks) from input images using encoder–decoder CNN architectures (U-Net variants).

The workflow includes:
- Data loading + preprocessing
- Train/test split
- Model training and validation
- Visualization of predictions (image / ground-truth mask / predicted mask)
- Metric evaluation (Mean IoU, Precision, Recall, F1)

## Dataset & preprocessing (as used in the notebooks)
- Images are handled at `128 × 128` with a single channel.
- The “normal” class is excluded because it does not have masks (binary segmentation).

Shapes reported:
- `X` shape: `(647, 128, 128, 1)`
- `y` shape: `(647, 128, 128, 1)`
- `X_train`: `(582, 128, 128, 1)`, `y_train`: `(582, 128, 128, 1)`
- `X_test`: `(65, 128, 128, 1)`, `y_test`: `(65, 128, 128, 1)`

## Models

### U-Net
A baseline U-Net with an encoder (Conv + MaxPool), bottleneck, and decoder (Conv2DTranspose + skip connections).  
Trained for binary segmentation.

### Attention U-Net
An Attention U-Net variant that uses attention-gated skip connections to emphasize relevant spatial features.  
Configured with Adam (learning rate `1e-4`) and binary cross-entropy; model size reported as **31,729,797 trainable parameters**.

## Results (test set)
Predictions are thresholded at **0.5**, then Mean IoU, Precision, Recall, and F1 are computed.

| Model | Mean IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| U-Net | 0.7963537 | 0.777 | 0.783 | 0.780 |
| Attention U-Net | 0.8156551 | 0.791 | 0.812 | 0.801 |

### Improvement (Attention U-Net − U-Net)

| Metric | Δ |
|---|---:|
| Mean IoU | +0.0193014 |
| Precision | +0.014 |
| Recall | +0.029 |
| F1 | +0.021 |

**Observation:** Attention U-Net improves all metrics, with the largest gain in Recall.

## How to run
1. Open the notebook for the model you want to run (U-Net or Attention U-Net).
2. Run cells top-to-bottom (preprocessing → training → evaluation).
3. Use the evaluation cell output to reproduce the metrics above.
