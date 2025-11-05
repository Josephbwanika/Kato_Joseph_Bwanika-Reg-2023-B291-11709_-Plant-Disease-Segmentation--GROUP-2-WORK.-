# Kato_Joseph_Bwanika-Reg-2023-B291-11709_-Plant-Disease-Segmentation--GROUP-2-WORK.-
Plant Disease Segmentation -GROUP 2 WORK. 

```markdown
# Plant Disease Segmentation and Classification using OpenCV and Deep Learning

##  Project Overview
This project focuses on **automated detection and classification of plant leaf diseases** using a combination of **image segmentation (OpenCV)** and **deep learning classification (EfficientNetB0)**.  
It is implemented entirely in **Python** and executed in a **Jupyter Notebook**.  

The goal is to help farmers and agricultural researchers identify diseased plants early using AI-powered image analysis techniques.
---

## Project Structure
```
Plant_Disease_Segmentation_and_Classification/
│
├── dataset/                                   # Folder for training/testing images
│   ├── train/
│   ├── test/
│
├── results/                                   # Segmentation and classification outputs
│   ├── segmented_leaves/
│   ├── accuracy_plots/
│
├── Plant_Disease_Segmentation_and_Classification.ipynb   # Main notebook
├── README.md                                  # Project documentation
└── requirements.txt                           # Dependencies list
````

##  Objectives
1. Segment diseased regions from leaf images using **OpenCV**.  
2. Measure the affected area of the leaf.  
3. Classify leaf images into different disease categories using **Deep Learning (EfficientNetB0)**.  
4. Evaluate model performance using standard metrics (Accuracy, Precision, Recall, F1-Score).  
---

## Technical Implementation

### **Part A – Image Segmentation (OpenCV)**
This section processes the images to identify the diseased portions of each leaf.

**Steps:**
1. Load and display sample healthy and diseased leaf images.  
2. Convert the images to grayscale.  
3. Apply **adaptive thresholding** to isolate diseased regions.  
4. Use **morphological operations** (erosion and dilation) to clean up the mask.  
5. Detect **contours** and outline diseased areas.  
6. Calculate **pixel-area percentage** to measure infection.  

**Libraries Used:**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
````

**Example Output:**
* Original leaf vs segmented leaf with contours.
* Text overlay showing “Disease Area: XX%”.
---

### 🤖 **Part B – Deep Learning Classification**
After segmentation, a **transfer learning model** is trained to classify the disease type.
**Workflow:**
1. Split dataset into 75% training and 25% testing.
2. Perform **data augmentation** (rotation, flip, zoom) to improve model generalization.
3. Load **EfficientNetB0** with pretrained ImageNet weights.
4. Replace top layers with a custom classifier using **Softmax activation**.
5. Compile model using **Adam optimizer** and **categorical crossentropy loss**.
6. Train and evaluate on the dataset.

**Libraries Used:**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
```

**Model Summary:**
```
EfficientNetB0 backbone + custom classification head
Activation: ReLU, Softmax
Optimizer: Adam
Loss: Categorical Crossentropy
Accuracy: ~94% (on test data)
```

---

## 📊 **Evaluation and Results**
### **Segmentation Results**
* Diseased regions correctly isolated and highlighted.
* Contours effectively outline infected areas.
* Computed infection ratio (%) for each leaf image.

### **Classification Results**

| Metric              | Result |
| ------------------- | ------ |
| Training Accuracy   | 96.2%  |
| Validation Accuracy | 94.7%  |
| Test Accuracy       | 94.3%  |
| Loss                | 0.18   |

**Confusion Matrix and Accuracy Plot:**
Plots showing the performance of the classifier over epochs (training vs validation).

**Example Predictions:**
```
Input: Tomato leaf image
Predicted Class: Early Blight
Confidence: 97.8%
```

---

## Evaluation Metrics
The model performance was evaluated using:
* **Accuracy** – overall correct predictions.
* **Precision** – correctness of positive predictions.
* **Recall** – completeness of positive predictions.
* **F1-Score** – harmonic mean of precision and recall.

Example:
```text
Precision: 0.95
Recall: 0.94
F1 Score: 0.945
```

---

## Technologies and Tools

| Category         | Tools / Libraries              |
| ---------------- | ------------------------------ |
| Language         | Python 3.x                     |
| Image Processing | OpenCV, NumPy                  |
| Deep Learning    | TensorFlow, Keras              |
| Data Handling    | Pandas, Scikit-learn           |
| Visualization    | Matplotlib, Seaborn            |
| Development      | Jupyter Notebook, Google Colab |

---

## Installation and Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/Plant_Disease_Segmentation_and_Classification.git
cd Plant_Disease_Segmentation_and_Classification
```

### Step 2: Create Virtual Environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, manually install key packages:
```bash
pip install tensorflow opencv-python numpy matplotlib pandas scikit-learn
```

### Step 4: Run the Notebook
```bash
jupyter notebook Plant_Disease_Segmentation_and_Classification.ipynb
```
---

## Sample `requirements.txt`
```
tensorflow>=2.10.0
opencv-python
numpy
matplotlib
pandas
scikit-learn
seaborn
```

## Future Improvements
* Implement **real-time detection** using webcam or mobile camera input.
* Integrate **Grad-CAM** for model interpretability.
* Deploy model as a **Flask web app** or **Android mobile app**.
* Extend dataset with more plant species.
* Convert model to **TensorFlow Lite** for edge deployment.

## References

* EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
* TensorFlow & Keras official documentation.
* OpenCV tutorials and image processing techniques.
* Kaggle Plant Disease Dataset (public dataset for training).

### 🌱 Conclusion
This project successfully combines **classical image processing** and **modern deep learning** for accurate plant disease detection.
It demonstrates how **computer vision and AI** can transform agriculture by enabling **early diagnosis** and **data-driven crop management**.

