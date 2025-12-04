# Composite-Material-Defect-Detection-System
A machine learning system combining diffusion models and zero-shot learning for automated defect detection and classification in composite materials.

---
## 📘 Overview

This system provides an end-to-end pipeline for detecting and classifying defects in composite materials using advanced machine learning techniques.  
The core innovation lies in:

- **Diffusion models** for synthetic data generation  
- **Zero-shot learning** for identifying *unseen* defect types  

This enables robust performance even under **limited labeled data**.

---

## ✨ Key Features

- **Diffusion-based Data Augmentation**  
  Generates realistic synthetic defect features using a conditional diffusion model.

- **Zero-shot Learning**  
  Classifies previously unseen defect types with semantic attribute embeddings.

- **Multi-model Comparison**  
  Includes Neural Network, Random Forest, and Gradient Boosting classifiers.

- **Comprehensive Visualization**  
  Training curves, performance plots, and t-SNE feature maps.

- **End-to-end Pipeline**  
  Automates data generation, augmentation, training, evaluation, and reporting.

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- NumPy  
- Matplotlib  
- Scikit-learn

---

## ▶️ Usage

Run the main pipeline:

```bash
python main.py
```

* The system will automatically execute the full workflow:

- **Framework initialization**

- **Data augmentation with diffusion model**

- **Model training**

- **Zero-shot evaluation on unseen defect classes**

- **Visualization output**

---

## ⚙️ Technical Details

### Diffusion Model

Implements a **conditional diffusion model** with cosine noise scheduling.

**Key parameters:**

- 1000 diffusion steps  
- Cosine β scheduling (β ∈ [0.0001, 0.02])  
- Conditioned on defect attribute vectors  

### Zero-shot Learning

Based on a **visual-semantic joint embedding space**:

- Multi-head self-attention visual encoder  
- Attribute-weighted semantic embeddings  
- Projection into a **128-dimensional** joint feature space

---

> 📄 The research paper for this project, **"Composite Material Surface Microscopic Defect Detection and Classification Combining Diffusion Models and Zero-shot Learning"**, has been accepted by *Scientific Reports*.

