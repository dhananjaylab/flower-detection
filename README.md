# 🌸 Flower Image Classification using Convolutional Neural Networks (CNN)

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-green.svg)]()

## 🌟 Project Overview

This project showcases a complete machine learning pipeline for **5-class flower image classification**. It uses a custom-built Convolutional Neural Network (CNN) implemented with **TensorFlow/Keras**. The pipeline emphasizes automated data retrieval, robust data augmentation techniques, and comprehensive performance evaluation.

### Key Achievements
* **Automated Data Retrieval** using the Kaggle API for maximum reproducibility.
* Implemented **Image Data Augmentation** (rotation, shifts, flips) to prevent overfitting.
* Trained a deep CNN architecture achieving competitive validation accuracy.

### Final Performance Metrics (Epoch 64)
| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **~74.61%** |
| Validation Loss | ~0.6917 |

***

## 🛠 CNN Architecture

The model is a sequential CNN, consisting of two major blocks (Conv + Pool) followed by three fully connected layers with a Dropout regularization layer.

| Layer Type | Filters/Neurons | Output Shape | Parameters | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Conv2D** | 64 | (128, 128, 64) | 1,792 | ReLU |
| **MaxPooling2D** | - | (64, 64, 64) | 0 | - |
| **Conv2D Stack** | 128 (x3) | (64, 64, 128) | 369,024 | ReLU |
| **MaxPooling2D** | - | (32, 32, 128) | 0 | - |
| **Flatten** | - | (131072) | 0 | - |
| **Dense** | 128, 64 | (128), (64) | 16,777,344 | ReLU |
| **Dropout** | Rate 0.25 | (64) | 0 | - |
| **Dense (Output)** | 5 | (5) | 325 | **Softmax** |

**Total Trainable Parameters: 17,156,741**

The full model architecture diagram is saved as: **`model_architecture.png`** 

[Image of Convolutional Neural Network Architecture Diagram]


***

## 📈 Evaluation and Results

The notebook generates the following critical visualizations to assess model performance:

1.  **Model Accuracy & Loss History:** Plots show the training vs. validation accuracy and loss over 64 epochs, confirming stable learning and generalization.
2.  **Confusion Matrix:** A heatmap visualization detailing correct vs. incorrect predictions across all 5 flower classes.
3.  **Prediction Grid:** A $6 \times 6$ grid of test images, visually displaying the model's prediction vs. the true label (green for correct, red for incorrect).

***

## 🚀 Setup and Execution (Reproducibility)

### 1. Environment Setup

Clone the repository and install the dependencies listed in `requirements.txt`:

```bash
git clone <your-repo-url>
cd dhananjaylab-flower-detection
pip install -r requirements.txt
```

### 2. Dataset Setup (Automated via Kaggle API)

This project automatically downloads the **`alxmamaev/flowers-recognition`** dataset directly from Kaggle using the Kaggle API.

**Prerequisite:** For the Kaggle API to authenticate and download the dataset:
1.  You must have a Kaggle account.
2.  Generate a Kaggle API token (`kaggle.json`) from your account settings.
3.  Place the `kaggle.json` file in your home directory under a `.kaggle` folder:
    * **Linux/macOS:** `~/.kaggle/kaggle.json`
    * **Windows:** `C:\Users\<Username>\.kaggle\kaggle.json`

The notebook will handle the download, extraction, and file organization into the required `Image_CLF_Datasets/flowers` structure.