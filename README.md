# 🌿 PlantClassification

**Leaf Image-Based Plant Classification**  
This repository contains the public version of a plant classification project based on leaf images. The web application components are not included.
 * Web-app not included
---

## 🔁 Project Workflow

1. **Image Reading**
2. **Image Processing & Feature Extraction** – Extract various features from each image to create a feature vector.
3. **Export to CSV** – Feature vectors are saved into a structured CSV file.
4. **Data Visualization** – Visualize features for exploration and understanding.
5. **Model Building & Comparison** – Train models, compare performance, and select the best.
6. **Model Evaluation** – Evaluate model robustness and accuracy.

---

## 📁 Folder Structure

Your input folder should contain subfolders named after plant classes. For example:

  Plant_Image/<br/>
  ├── Class_A<br/>
  ├── Class_B<br/>
  ├── Class_C<br/>

Feature extraction will automatically label samples based on these subfolder names.

---

## 🐍 Python Files Overview

### 🔍 Feature Extraction
- `Extract/_ExtractionToCSV.py`  
  → Main pipeline for feature extraction. Input is a folder path; output is a CSV file.  
- `Extract/Feature.py`  
  → Contains all feature extraction functions.

### ⚙️ Utilities
- `Util/BG_remove.py`  
  → Removes background using segmentation. Sample usage in `Image_Segmentation.py`.  
- `Util/Image_IO.py`  
  → Handles image I/O and utilities like file format conversion.

### 🖼️ Image Segmentation
- `Image_Segmentation.py`  
  → Main script to perform background removal using utility functions.

### 🧠 Model Creation & Testing
- `MakeModel.py`  
  → Builds and compares models. Includes automated model fine-tuning and evaluation.  
- `NoiseAdding.py`  
  → Adds artificial noise to images to test model robustness.  
- `RobustTest.py`  
  → Evaluates model performance on noisy datasets.

### 📊 Visualization
- `Visual.py`  
  → Generates visualizations such as box plots to understand data distribution.
