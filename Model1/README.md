# 🧠 Alzheimer's MRI Classification using VGG16 (PyTorch)

## 📌 Overview
This project builds a **Deep Learning pipeline** to classify brain MRI scans into different stages of Alzheimer's disease using **Transfer Learning with VGG16** in PyTorch.  
The model is trained on an **augmented MRI dataset**, leveraging a pre-trained VGG16 network as a feature extractor, while fine-tuning the classifier layers for Alzheimer’s detection.

---

## 📂 Dataset
- **Source:** Augmented MRI brain scans  
- **Classes:** 4 categories (replace with your dataset’s class names)  
  - `MildDemented`  
  - `ModerateDemented`  
  - `NonDemented`  
  - `VeryMildDemented`  
- **Splits:**
  - 70% Training  
  - 15% Validation  
  - 15% Testing  

Dataset splitting is done using the [`split-folders`](https://pypi.org/project/split-folders/) library.

---

## ⚙️ Project Workflow

### 1️⃣ Dataset Preparation
- Organized dataset into train/val/test splits.
- Applied preprocessing:
  - Resizing images to **224×224** (VGG16 input size).
  - Normalizing pixel values to `[0,1]`.

### 2️⃣ Model Setup (VGG16)
- Loaded **pre-trained VGG16** from ImageNet.
- **Froze convolutional layers** to retain feature extraction.
- Modified classifier head:
  ```python
  vgg16.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(25088, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, num_classes)
  )
- Loss Function: CrossEntropyLoss

- Optimizer: Adam (learning rate = 1e-4)

### 3️⃣ Training

- **Custom training loop for 20 epochs:**

 - Forward pass → Loss computation → Backpropagation → Optimizer step

 - Tracked training & validation loss/accuracy

- **Metrics stored in a history dictionary:**
  - history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
    }

### 4️⃣ Validation

- After each epoch:

 - Computed validation accuracy & loss.

 - Printed metrics for progress tracking.





#### 🚀 How to Run
1. Clone this repository
 -  git clone https://github.com/PRADNESH-PATIL/Alzheimers-MRI-Classification.git
 

2. Install dependencies

 -  Make sure you have Python 3.11+ installed.

 -  Install all required packages:
    - pip install -r requirements.txt

3. Prepare the dataset

Place your augmented MRI dataset in a folder, e.g.:

 -  Dataset-Link: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

 -  Dataset-MRI-img/AugmentedAlzheimerDataset


Update the dataset path in the script/notebook:

 -  input_folder  = r"path of the file"
 -  output_folder = r"path split data need to be stored"

4. Split the dataset

Run the dataset split code:

 -  import splitfolders
 -  splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.15, 0.15))

5. Train the model

 -  If running via Jupyter Notebook → open and run cells sequentially.

 -  If converted to a script:

    -   python train.py

6. Evaluate the model

 -  After training, evaluate on validation/test set.

 -  Example metrics printed:

    -- Epoch 5/20 -> Train Loss: 0.4567, Train Acc: 0.8234, Val Loss: 0.5123, Val Acc: 0.8010




##### 📊 Results

 -  Training and validation accuracy/loss are stored in the history dictionary.

- **Training Accuracy:** 98.90%  
- **Validation Accuracy:** 95.31%  
- **Test Accuracy:** 95.55% ✅

![Confusion Matrix](assets/images/confusion-matrix.png)
![Learning Curves Plot](assets/images/plot-learning-curves.png)
![ROC Curve](assets/images/roc.png)
![Classification Report](assets\images\classification_report.png)

###### 🛠️ Tech Stack

Programming Language: Python

Deep Learning Framework: PyTorch

Model: VGG16 (Transfer Learning)

Libraries: split-folders, torchvision, tqdm, torchsummary



👨‍💻 Author

Pradnesh Patil 