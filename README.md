---
title: Face Mask Detection System
emoji: 😷
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: apache-2.0
base_model: vgg16
tags:
  - face-mask-detection
  - computer-vision
  - pytorch
  - vgg16
  - gradio
github_repo: wsmaisys/FaceMask-Detection-CNNModel-VGG16
---

# Face Mask Detection System

A sophisticated real-time face mask detection system leveraging deep learning and computer vision. This project combines the power of transfer learning using VGG16 architecture with custom classification layers to accurately detect and classify face mask usage.

## 🔗 Project Links

- 🚀 **Live Demo**: [Face Mask Detector on Hugging Face Spaces](https://huggingface.co/spaces/wassim-ansari2088/Face-Mask-Detector-Vision)
- 💻 **GitHub Repository**: [FaceMask-Detection-CNNModel-VGG16](https://github.com/wsmaisys/FaceMask-Detection-CNNModel-VGG16)
- 🤗 **Hugging Face Space**: [wassim-ansari2088/Face-Mask-Detector-Vision](https://huggingface.co/spaces/wassim-ansari2088/Face-Mask-Detector-Vision)

## Project Overview

This system can detect and classify face mask usage in real-time, categorizing it into three classes:
- ✅ Properly wearing mask
- ⚠️ Incorrectly wearing mask
- ❌ Not wearing mask

### Key Features

- 😷 Real-time face mask detection
- 🎯 Three-way classification with confidence scores
- 📸 Webcam support for live detection
- 🖥️ User-friendly Gradio interface with neon theme
- 🚀 Optimized for performance using PyTorch
- 🔍 Robust face detection using Haar Cascade

## Technical Architecture

### 1. Model Architecture: Why VGG16?

The system leverages VGG16 as the backbone for several compelling reasons:

1. **Proven Feature Extraction**:
   - VGG16's deep architecture (16 layers) excels at hierarchical feature learning
   - Strong feature representations from simple edges to complex patterns
   - Particularly effective for facial feature detection

2. **Transfer Learning Benefits**:
   - Pre-trained on ImageNet (1.4M images)
   - Robust low-level feature extraction
   - Significantly reduces training time and data requirements

3. **Architecture Adaptability**:
   - Simple, uniform architecture makes modification straightforward
   - Easy to freeze/unfreeze layers for fine-tuning
   - Excellent balance of depth and computational efficiency

### 2. Custom Classification Layers

We've designed a custom classifier head optimized for mask detection:

```
VGG16 (Frozen) → Flatten → Dense Architecture
```

Layer Structure:
1. **Input Processing**:
   - Input: 224x224x3 RGB images
   - VGG16 backbone (frozen): outputs 25088 features

2. **Custom Classifier**:
   ```
   Flatten (25088)
   └── Dense(25088 → 256) + ReLU + Dropout(0.33)
       └── Dense(256 → 128) + ReLU + Dropout(0.33)
           └── Dense(128 → 3) + Softmax
   ```

3. **Design Rationale**:
   - Gradual dimension reduction prevents information bottleneck
   - Dropout layers (0.33) prevent overfitting
   - ReLU activations maintain non-linearity
   - Final Softmax for probability distribution

## Deployment Details

### Hugging Face Spaces Integration

The project is deployed on Hugging Face Spaces, offering:
- 🌐 Easy accessibility through web browser
- � No local setup required
- 📊 Real-time processing capabilities

### Real-Time Detection Limitations

Important note about Hugging Face Spaces deployment:

1. **Manual Trigger Requirement**:
   - Continuous real-time detection is not available
   - Each detection requires a manual button click
   - This is a platform limitation, not a technical one

2. **Why This Limitation?**:
   - Hugging Face Spaces has resource constraints
   - Continuous video processing would be resource-intensive
   - Manual triggering ensures stable performance

3. **Local Deployment Alternative**:
   - For continuous real-time detection
   - Clone repository and run locally
   - No button clicks needed for local deployment

## Implementation Details

### Face Detection Pipeline

1. **Preprocessing**:
   - RGB conversion for consistency
   - Haar Cascade for face detection
   - Dynamic bounding box computation

2. **Inference**:
   - Face region extraction and resizing (224x224)
   - PyTorch tensor conversion
   - Model prediction with confidence scores

3. **Visualization**:
   - Color-coded bounding boxes:
     - Green: Proper mask
     - Orange: Incorrect mask
     - Red: No mask
   - Confidence score display

## Dependencies

- torch & torchvision: Deep learning framework
- opencv-python-headless: Computer vision operations
- gradio: UI framework
- numpy: Numerical computations

## Setup and Usage

1. **Interface Access**:
   - Visit the [Hugging Face Space](https://huggingface.co/spaces/wassim-ansari2088/Face-Mask-Detector-Vision)
   - Allow camera access when prompted

2. **Detection Process**:
   - Click "Enable Webcam"
   - Position face(s) in view
   - Click "Detect Masks"
   - View results with bounding boxes and confidence scores

## Author

[Waseem M Ansari](https://huggingface.co/wassim-ansari2088)

## License

This project is open-source and available under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.