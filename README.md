# Plant Disease Segmentation and Classification (GROUP 2 WORK)

**Author:** Kato Joseph Bwanika  
**Reg. No:** 2023-B291-11709  
**Course:** Artificial Intelligence – UMU University  

---

## Project Overview
This project implements **automated plant leaf disease detection** using:
- **OpenCV** for image segmentation to isolate diseased areas.  
- **Deep Learning (EfficientNetB0)** for classifying different disease types.  

The system helps identify early-stage plant infections using **AI-powered image analysis**.

---

## Workflow Summary

### **Part A: Image Segmentation (OpenCV)**
- Convert leaf images to grayscale.  
- Apply **adaptive thresholding** and **morphological operations**.  
- Detect and outline diseased regions.  
- Calculate infection percentage based on pixel area.

### **Part B: Disease Classification (Deep Learning)**
- Dataset split: **75% Training / 25% Testing**.  
- Used **EfficientNetB0 (Transfer Learning)**.  
- Applied **data augmentation** to prevent overfitting.  
- Achieved **~94% accuracy** on test data.

---

## Results
| Metric | Value |
|--------|--------|
| Training Accuracy | 96.2% |
| Validation Accuracy | 94.7% |
| Test Accuracy | 94.3% |
| Loss | 0.18 |

**Example Prediction:**  
`Input: Tomato leaf → Predicted: Early Blight (Confidence: 97.8%)`

---

## Tools & Technologies
- **Python 3.x**  
- **OpenCV**, **NumPy**, **Matplotlib**  
- **TensorFlow**, **Keras**, **Scikit-learn**  
- **Jupyter Notebook / Google Colab**

---

## How to Run
```bash
git clone https://github.com/<your-username>/Plant_Disease_Segmentation_and_Classification.git
cd Plant_Disease_Segmentation_and_Classification
pip install -r requirements.txt
jupyter notebook Plant_Disease_Segmentation_and_Classification.ipynb


## Future Work

* Real-time detection via camera input.
* Model deployment using Flask or TensorFlow Lite.
* Grad-CAM visualization for model explainability.

---

## References
* TensorFlow & Keras Documentation
* OpenCV Tutorials
* EfficientNet Research Paper
* Kaggle Plant Disease Dataset
---

### Conclusion
This project demonstrates how **AI and computer vision** can be applied to **agriculture** for early and accurate **plant disease detection**, supporting smarter and data-driven crop management.
---

