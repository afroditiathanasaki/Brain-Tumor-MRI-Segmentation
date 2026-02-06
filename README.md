# Brain Tumor MRI Segmentation

This repository contains the code developed as part of my Integrated Master’s thesis in Applied Mathematics at the National Technical University of Athens (NTUA).

The project focuses on **brain tumor segmentation from MRI scans** using deep learning techniques, with emphasis on **medical image preprocessing, data pipelines, and model training** in a realistic clinical data setting.

---

## Project Overview

Accurate segmentation of brain tumors in MRI images is a critical task in medical imaging, supporting diagnosis, treatment planning, and monitoring of disease progression.

In this project, I implemented an end-to-end workflow for:
- Preprocessing 3D brain MRI volumes
- Preparing data loaders for training and evaluation
- Training deep learning models for semantic segmentation
- Evaluating and visualizing segmentation results

The pipeline is implemented using **Python** and the **MONAI framework**, which is widely used in medical imaging and healthcare AI applications.

---
## Segmentation Models

As part of this work, multiple deep learning architectures were explored and compared for 3D brain tumor semantic segmentation.

The following models were implemented and evaluated:

- **U-Net**  
  A widely used baseline architecture for medical image segmentation.

- **SegResNet**  
  A residual convolutional network designed for efficient and stable training on volumetric medical data.

- **VNet**  
  A fully convolutional 3D architecture optimized for volumetric segmentation tasks.

- **Swin UNETR**  
  A hybrid transformer–CNN architecture that leverages hierarchical self-attention for global context modeling in 3D medical images.

All models were trained using the same preprocessing pipeline and evaluated under identical conditions to ensure fair comparison.


## Repository Structure

```text
Brain-Tumor-MRI-Segmentation/
│
├── preprocess.py        # Data loading and preprocessing pipeline
├── training.py          # Model definition and training loop
├── utilities.py         # Helper functions (metrics, visualization, utilities)
├── testing.ipynb        # Notebook for evaluation and qualitative results
├── requirements.txt     # Python dependencies
└── README.md
