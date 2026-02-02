# Composite Material Surface Microscopic Defect Detection and Classification (Diffusion-ZSL)

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Paper: Nature Scientific Reports](https://img.shields.io/badge/Paper-Scientific%20Reports-blue)](https://doi.org/10.1038/s41598-025-29871-w)

[cite_start]This repository implements a joint framework for composite material defect detection and classification by deeply integrating **Diffusion Models** and **Zero-shot Learning (ZSL)**[cite: 5, 54]. [cite_start]This system addresses the critical industrial challenges of limited labeled data and the emergence of previously unseen defect types during manufacturing[cite: 6, 37].

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Technical Details](#-technical-details)
    * [Defect Attributes](#defect-attributes)
    * [Loss Function Weights](#loss-function-weights)
7. [Experimental Results & Visualization](#-experimental-results--visualization)
8. [Citation](#-citation)
9. [License](#-license)

---

## 📘 Project Overview

[cite_start]Traditional deep learning models for surface inspection are limited by their dependence on massive labeled datasets and their inability to generalize to novel defect categories[cite: 23, 31]. 

[cite_start]This project constructs a **Dual-path Collaborative Architecture**[cite: 7, 56]:
* [cite_start]**Diffusion Path**: Utilizes an improved Denoising Diffusion Probabilistic Model (DDPM) to learn the potential distribution of defects and generate high-quality synthetic features for data augmentation[cite: 8, 56, 120].
* [cite_start]**Zero-shot Path**: Realizes cross-modal knowledge transfer through a vision-semantic joint embedding space, enabling the recognition of "unseen" defect categories without any training samples[cite: 8, 35, 61].

### Core Performance Metrics
* [cite_start]**Generation Quality**: Achieves a Fréchet Inception Distance (FID) of **18.2**, significantly outperforming traditional GANs (42.8)[cite: 352, 358, 605].
* [cite_start]**Detection Accuracy**: Reaches a **0.777 mAP@50** on the NEU-DET benchmark[cite: 354, 365, 640].
* [cite_start]**Economic Value**: Achieving an F1-score of **0.692** with zero labels, equivalent to traditional supervised learning with 500 labels, resulting in **100% labeling cost reduction**[cite: 450, 469, 641].
* [cite_start]**Real-time Efficiency**: Average inference time of **140.5ms**, meeting industrial real-time requirements[cite: 612, 640].

---

## ✨ Key Features

* [cite_start]**Conditional Diffusion Generation**: Integrates category-specific constraints and spatial masks via cross-attention to control the synthesis of defect patterns[cite: 46, 131].
* **Attribute-driven Reasoning**: Decomposes visual characteristics into 18 weighted semantic descriptors (e.g., shape, texture, boundary), enabling knowledge transfer to novel classes.
* [cite_start]**Feature Space Alignment**: Employs **Maximum Mean Discrepancy (MMD) loss** to ensure distributional consistency between generative and discriminative pathways[cite: 47, 205, 242].
* [cite_start]**Cross-Material Adaptability**: Demonstrates high generalization across carbon fiber, glass fiber, and aramid fiber composites[cite: 317, 473, 524].

---

## ⚙️ System Architecture

[cite_start]The framework consists of four integrated modules[cite: 54, 108]:

| Module | Implementation Details | Key Parameters |
| :--- | :--- | :--- |
| **Diffusion Module** | U-Net + Improved DDPM | 1000 steps, Cosine $\beta$ schedule [0.0001, 0.02] |
| **Zero-shot Module** | ViT-B/16 + BERT-base | 128D Joint Embedding, 8-head Attention |
| **Fusion Module** | Multi-modal Attention Fusion | Cross-attention + Gating at 32x32 resolution |
| **Classification Head** | Neural Network / RF / GB | Hidden layers (128, 64, 32) for MLP |

---

## 🔧 Installation

### Prerequisites
* Python 3.8+
* NumPy
* Matplotlib
* Scikit-learn
* PyTorch 1.12+ (Recommended for deep learning backbones)

## ⚙️ Technical Details

### Defect Attributes
[cite_start]Defects are characterized by weighted attributes to bridge the vision-semantic gap[cite: 178, 191]. These attributes represent visual and structural properties defined in `defect_attributes.py`. Example weights for selected categories include:

* **Crack**: Linear shape (0.18), Aligned orientation (0.13), Edge sharpness (0.10).
* **Inclusion**: Circular shape (0.17), Embedded depth (0.15), Contrast intensity (0.13).
* **Delamination (Unseen)**: Planar shape (0.16), Layered structure (0.15), Interface location (0.14).

### Loss Function Weights
[cite_start]The total objective function balances generation quality, classification performance, and feature space alignment[cite: 251, 252]. The weighted scheme is defined as:  
[cite_start]$L_{total} = \lambda_1 L_{diff} + \lambda_2 L_{cls} + \lambda_3 L_{align} + \lambda_4 L_{reg}$[cite: 253].

Optimal weights configured in `config.py` are:
* **$\lambda_1$ (Diffusion Loss)**: 1.0
* **$\lambda_2$ (Classification Loss)**: 1.5
* **$\lambda_3$ (MMD Alignment Loss)**: 0.3
* **$\lambda_4$ (Regularization)**: 0.01
