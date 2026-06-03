# **MNIST Digit Classifier**

## **📋 Assignment Overview**

**Objective**: An AI-based image processing program designed to classify handwritten digits (0-9). This implementation focuses on end-to-end machine learning: from image normalization and feature extraction to predictive inference.

**Key Results**:

- **Test Accuracy**: 99.11%
- **Training Time**: ~45 Seconds (Tested on CPU)
- **Model Size**: ~1.6 MB (State Dictionary)

---

## **🚀 Quick Start**

### **Prerequisites**

- Python 3.8+
- pip package manager
- Stable internet connection (for initial dataset download)

### **Setup & Execution**

Run the following commands in your terminal/command prompt:

```python
# 1. Install dependencies
pip install torch torchvision matplotlib

# 2. Run the classifier
python main.py
```

---

## **📁 Project Structure**

```
MNIST-Digit-Classifier/
├── main.py              # Complete source code (Model, Training, & Evaluation)
├── README.md            # Execution guide and overview
├── APPROACH.txt         # Deep dive into technical design decisions
├── model.pth            # Saved model weights (Generated after run)
└── predictions.png      # Visualization of sample predictions (Generated after run)
```

---

## **🧠 Architecture & Dimensions**

The model utilizes a custom Convolutional Neural Network (CNN) to capture spatial hierarchies in the 28x28 input images.

```
Input (28x28) 
  → Conv2D(1→32, kernel=3) → ReLU → MaxPool(2)  [Output: 32x14x14]
  → Conv2D(32→64, kernel=3) → ReLU → MaxPool(2) [Output: 64x7x7]
  → Flatten()                                   [Output: 3136]
  → Linear(3136→128) → ReLU                     [Output: 128]
  → Linear(128→10)                              [Output: 10]
```

### **Training Configuration**

- **Optimizer**: Adam (Learning Rate: 0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 3
- **Reproducibility**: `torch.manual_seed(42)`

---

## **📊 Performance Metrics**

| **Metric** | **Value** |
| --- | --- |
| Final Test Accuracy | **99.11%** |
| Training Time | ~45 seconds (Intel i7 / CPU) |
| Model Parameters | 421,642 |
| Device Used | CPU (Optimization: `num_workers=0`) |

---

## **🎯 Features Implemented**

- ✅ **Image Processing**: Automated normalization and tensor transformation.
- ✅ **Custom CNN**: Built from scratch using PyTorch `nn.Module`.
- ✅ **Metrics Tracking**: Real-time accuracy and loss logging per epoch.
- ✅ **Visual Inference**: Generates `predictions.png` showing predicted vs. true labels.
- ✅ **Confidence Scoring**: Uses Softmax probabilities to show model certainty.
- ✅ **Persistence**: Saves model state for future deployment without retraining.

---

## **📝 Expected Terminal Output**

```
MNIST Digit Classifier
========================================

1. Loading MNIST dataset...
   Training: 60000 images
   Testing:  10000 images

2. Building CNN model...
   Model parameters: 421,642

3. Training model (3 epochs)...
   Epoch 1: 95.83% accuracy
   Epoch 2: 98.71% accuracy
   Epoch 3: 99.11% accuracy

4. Testing on unseen data...
   Test Accuracy: 99.11%

5. Making sample predictions...

6. Confidence scores:
   Image 0: 7 (100.0% confidence)
   Image 1: 2 (100.0% confidence)
   Image 2: 1 (100.0% confidence)
   Image 3: 0 (100.0% confidence)
   Image 4: 4 (100.0% confidence)

7. Saving model...
   Saved as 'model.pth'

========================================
SUMMARY
========================================
Final test accuracy: 99.11%
Files saved: predictions.png, model.pth
========================================
```

---

## **🔍 Technical Notes**

- **Design Choice**: 2 Convolutional layers were selected to balance high accuracy with low computational cost, making it ideal for CPU execution.
- **Normalization**: Used Mean (0.1307) and Std (0.3081) to ensure input data is centered, preventing gradient saturation.
- **Inference**: During testing, the model uses `model.eval()` and `torch.no_grad()` to reduce memory overhead and disable dropout layers.