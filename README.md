#      Facial-Emotion-Recognition-Kaggle-FER-2013
   **Pixels ➡️ Features ➡️ Emotions!** 🖼️🧠💗

💡A deep learning project for facial emotion recognition using **PyTorch** and a custom **Convolutional Neural Network (CNN)** trained from scratch on the **FER2013 dataset** from Kaggle. The model achieves a best observed test accuracy of approximately **69.04%**.


## What does it do?

The model looks at a facial image and predicts one of **7 emotions**:

* 😠 Angry
* 🤢 Disgust
* 😨 Fear
* 😄 Happy
* 😐 Neutral
* 😢 Sad
* 😲 Surprise


## 📦 Setup

### 1. Install the required packages

```bash
pip install torch torchvision matplotlib seaborn numpy scikit-learn
```

### 2. Prepare the FER2013 dataset 📁

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

`Image → Conv2D → BatchNorm → ReLU → MaxPool → Dropout → ... → Adaptive Average Pooling → Fully Connected → Emotion`

The network gradually learns facial features and patterns that help distinguish different expressions.

## ✨ Data Augmentation

Training images are randomly transformed to improve generalization:

* ↔️ Random horizontal flipping
* 🔄 Random rotations up to ±10°

## 🚀 Training

For each epoch, the model follows:

`Train 🏋️ → Evaluate 🧪 → Calculate Accuracy 📊 → Save Best Model 💾`




