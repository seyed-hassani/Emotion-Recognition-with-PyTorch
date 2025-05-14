# 🧠 Emotion Recognition with PyTorch

This project trains a deep learning model to recognize human emotions from grayscale facial images. Using transfer learning with a pre-trained ResNet-18 model, we classify facial expressions into seven categories: **Anger**, **Disgust**, **Fear**, **Happiness**, **Sadness**, **Surprise**, and **Neutral**.

---

## 📁 Dataset

- Grayscale images (`48x48`) with corresponding emotion labels  
- Stored in compressed `.parquet.gzip` format for optimized loading and memory usage  
- Labels are encoded as integers from 0 to 6, mapped as follows:

---

## 🧪 Emotions Categories

| Label | Emotion    |
|-------|------------|
| 0     | Anger      |
| 1     | Disgust    |
| 2     | Fear       |
| 3     | Happiness  |
| 4     | Sadness    |
| 5     | Surprise   |
| 6     | Neutral    |

---

## 🧰 Model Overview

- **Base Model**: ResNet-18 pretrained on ImageNet  
- **Fine-Tuning**:  
  - Replaced last FC layer with:  
    - `Linear(512 → 100)` + `ReLU`  
    - `Linear(100 → 7)` + `Softmax`  
- **Framework**: PyTorch  
- **Loss Function**: Cross Entropy Loss  
- **Optimizer**: Adam (learning rate: `1e-4`)  
- **Epochs**: 30  
- **Batch Size**: 128  

---

## 🎯 Evaluation

- **Validation Accuracy**: ~68.5%  
- **Validation Loss**: ~1.47  
- **Visualization**: Confusion matrix for class-wise performance  
- **Metric**: Accuracy on test set (🎯 must achieve ≥ **63%**)

---

## 🔍 Key Features

- Custom `Dataset` and `DataLoader` for grayscale image handling  
- Data normalization with computed `mean` and `std`  
- Augmentation: `RandomRotation` used during training  
- Final prediction export for downstream tasks  
- Easy visualization and diagnostics with plots and matrix  

---

## 🚀 How to Run

1. Clone the repository and upload the required `.parquet.gzip` files  
2. Install requirements:
   ```bash
   pip install torch torchvision pandas scikit-learn matplotlib
