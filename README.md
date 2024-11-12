

# Dehazing with Deep Learning

This project demonstrates a deep learning approach to dehazing images using a modified ResNet18 model. The network is designed to restore clarity to hazy images by learning to map hazy inputs to ground-truth (clear) outputs. This README provides an overview of the project structure, dataset preparation, model architecture, training procedure, and evaluation method.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Requirements](#requirements)
7. [Usage](#usage)
8. [Results](#results)

---

### Project Overview

The goal of this project is to develop a real-time, high-performance dehazing solution. Leveraging a pretrained ResNet18 backbone with additional convolutional and upsampling layers, this architecture can generate clear versions of hazy images with strong visual accuracy.

### Dataset

The dataset is expected to be structured as follows:

```
data/
├── GT # Ground Truth images (clear images)
└── hazy # Hazy images
```

Each subfolder should contain paired images with matching filenames in both `GT` and `hazy` folders. If the folders or files are not correctly formatted, errors may occur.

### Model Architecture

The core architecture extends the ResNet18 network by:
1. Removing its final fully connected layers to retain only feature extraction capabilities.
2. Adding additional convolutional layers for feature processing and image restoration.
3. Using upsampling layers to reconstruct the original image size.

The `DehazingResNet18` class encapsulates this architecture.

### Training

The training process involves:
1. **Loading the Dataset**: Images from the `GT` and `hazy` directories are loaded and preprocessed.
2. **Data Augmentation**: Applied transformations include resizing, normalization, and conversion to tensor.
3. **Loss Calculation**: Mean Squared Error (MSE) Loss measures the pixel-wise difference between dehazed and ground-truth images.
4. **Optimization**: The Adam optimizer is used to train the model.
5. **Training Loop**: The model trains for 30 epochs, with loss printed at regular intervals to monitor progress.

### Evaluation

After training, evaluation includes:
- **Visual Inspection**: Displaying hazy, ground-truth, and dehazed images for comparison.
- **Quantitative Metrics**: Using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) to measure the quality of the dehazed images.

### Requirements

Install the following libraries:
```bash
pip install torch torchvision pillow scikit-image matplotlib pandas
```

### Usage

1. **Prepare Data**: Place hazy and clear images in the `data/hazy` and `data/GT` folders, respectively.
2. **Run Training**:
   ```python
   python train.py
   ```
3. **Save Model**: The trained model's weights will be saved as `Dehazing.pth`.
4. **Evaluate**: Run the evaluation to visualize the dehazed output and calculate metrics.

### Results

The model restores image clarity effectively, achieving satisfactory PSNR and SSIM values. Below are some sample results showing the dehazing performance:

![Dehazed Output Sample](images/sample_output.png)


---

