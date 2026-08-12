#      Facial-Emotion-Recognition-Kaggle-FER-2013
   **Pixels ➡️ Features ➡️ Emotions!** 🖼️🧠💗

💡 A simple deep learning project for recognizing facial emotions from images using **PyTorch** and a custom **Convolutional Neural Network (CNN)**. 

## What does it do?

The model looks at a facial image and predicts one of **7 emotions**:

* 😠 Angry
* 🤢 Disgust
* 😨 Fear
* 😄 Happy
* 😐 Neutral
* 😢 Sad
* 😲 Surprise

The project uses the **FER2013 dataset** from Kaggle, where facial images are grayscale and resized to **48 × 48 pixels**.

## 📦 Setup

### 1. Install the required packages

```bash
pip install torch torchvision matplotlib seaborn numpy scikit-learn
```

### 3. Prepare the FER2013 dataset 📁

The dataset should contain separate `train` and `test` folders:

```text
archive/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## 🧠 Model

A custom CNN is built with:

`Image → Conv2D → ReLU → BatchNorm → MaxPool → Dropout → ... → Fully Connected → Emotion`

The network gradually learns facial features and patterns that help distinguish different expressions.

## ✨ Data Augmentation

Training images are randomly transformed to improve generalization:

* ↔️ Horizontal flipping
* 🔄 Small rotations
* 🖤 Grayscale normalization

## 🚀 Training

For each epoch, the model follows:

`Train 🏋️ → Evaluate 🧪 → Calculate Accuracy 📊 → Save Best Model 💾`
Training and test loss/accuracy are recorded to track performance.



