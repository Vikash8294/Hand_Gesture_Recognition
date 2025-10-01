# Hand Gesture Recognition

This repository is dedicated to recognizing hand gestures using machine learning and computer vision techniques. The primary language used is **Jupyter Notebook**, making it interactive and easy to experiment with.This project demonstrates American Sign Language (ASL) alphabet recognition using a deep learning approach with Convolutional Neural Networks (CNNs) in TensorFlow/Keras. The model is trained to classify hand gesture images representing the ASL alphabets (A-Z) and digits (0-9). The project covers the complete workflow: data loading, preprocessing, modeling, training, evaluation, and visualization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation & Requirements](#installation--requirements)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Data Visualization](#2-data-visualization)
  - [3. Label Encoding & Train-Test Split](#3-label-encoding--train-test-split)
  - [4. CNN Model Architecture](#4-cnn-model-architecture)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
  - [7. Confusion Matrix & Classification Report](#7-confusion-matrix--classification-report)
  - [8. Training History Visualization](#8-training-history-visualization)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Contact](#contact)

---

## Project Overview

- **Goal:** To accurately recognize ASL alphabet and digit hand gestures from RGB images (128x128 pixels).
- **Techniques:** Deep learning using CNNs, data augmentation, performance visualization, and evaluation metrics.
- **Result:** Achieves **~98% accuracy** on the test set with robust performance across all classes.

---

## Dataset

- **Directory:** `/content/drive/MyDrive/asl_dataset`
- **Structure:** Each class (0-9, a-z) has its own folder containing images of hand gestures.
- **Total Images:** 2515 (36 classes)
- **Image Size:** 128x128 pixels, 3 channels (RGB)

---

## Installation & Requirements

Ensure you have the following packages installed:

```bash
pip install numpy matplotlib scikit-learn tensorflow seaborn
```

- Python 3.7+
- TensorFlow/Keras
- Matplotlib
- Scikit-learn
- Seaborn

---

## Step-by-Step Workflow

### 1. Data Loading & Preprocessing

- Iterates through all class folders.
- Loads and resizes images to 128x128 pixels.
- Normalizes pixel values to [0, 1].
- Stores images and corresponding labels.

### 2. Data Visualization

- Randomly displays 15 sample images with labels using matplotlib for sanity check.

### 3. Label Encoding & Train-Test Split

- Encodes string labels (a-z, 0-9) into integer classes using `LabelEncoder`.
- Splits data: 80% training, 20% testing, stratified by class.

### 4. CNN Model Architecture

- **Layers:**
  - Conv2D + BatchNorm + MaxPooling (x4, increasing filters)
  - Flatten
  - Dense (512, ReLU) + Dropout
  - Output layer: Dense (36, softmax)
- **Optimizer:** Adam
- **Loss:** Sparse categorical crossentropy

### 5. Model Training

- Trained for 30 epochs, batch size 32.
- Training and validation accuracy/loss tracked.

### 6. Model Evaluation

- Uses `ImageDataGenerator` for test evaluation.
- Achieves **Test Accuracy: 97.69%**.

### 7. Confusion Matrix & Classification Report

- Generates confusion matrix heatmap for model predictions.
- Prints precision, recall, F1-score for each class.

### 8. Training History Visualization

- Plots training/validation accuracy and loss over epochs.

---

## Results

- **Test Accuracy:** **97.69%**
- **Classification Report:** High precision, recall, and F1-score for nearly all classes.
- **Confusion Matrix:** Model shows strong performance, minimal confusion among classes.

Example (partial) classification report:

```
              precision    recall  f1-score   support

           0       0.99      1.00      0.99        70
           1       1.00      0.94      0.97        70
           ...
           z       0.96      1.00      0.98        70

    accuracy                           0.98      2515
   macro avg       0.98      0.98      0.98      2515
weighted avg       0.98      0.98      0.98      2515
```

---

## Key Takeaways

- **End-to-End :** The project covers the full ML workflow for image classification.
- **Strong Generalization:** Achieves high test accuracy across diverse classes.
- **Visual Insights:** Training history and confusion matrix plots provide a clear understanding of model performance.
- **Modern Techniques:** Uses best practices (batch norm, dropout, stratified split).


## Contact

**Author:** [Vikash8294](https://github.com/Vikash8294)  
**Email:** vikashjaihind147@gmail.com

---
