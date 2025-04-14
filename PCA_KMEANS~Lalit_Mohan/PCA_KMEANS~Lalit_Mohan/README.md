# Project Report: PCA and K-Means Clustering for Fruit Image Analysis

**Author:** Lalit Mohan (B23EE1089)

## Introduction

This project focuses on dimensionality reduction using Principal Component Analysis (PCA) and clustering using the K-Means algorithm. The goal is to process and analyze image data efficiently, reduce high-dimensional feature space, and visualize clustering performance.

## Pipeline Overview

The complete machine learning pipeline is implemented and executed in Python using Scikit-learn and supporting libraries. Below are the major components:

### 1. Image Preprocessing

- Images from the Fruits-360 dataset were resized to 50x50 pixels for uniformity.
- Features were extracted from different representations: color histograms, raw pixel intensities, and CNN intermediate layer activations.
- Color histogram features were normalized and computed in the HSV color space.

### 2. Dimensionality Reduction

- PCA and Incremental PCA were used to reduce features to 2 or 3 components.
- A detailed analysis was performed to plot the relationship between `n_components` in PCA and K-Means silhouette scores, leading to the conclusion that the optimal `n_components` value was ~0.35 (explained variance threshold).
- PCA results were cached using Pickle for efficient reuse.

### 3. Clustering

- K-Means clustering was applied on PCA-transformed data.
- 136 clusters were used (based on label count).
- Clustering was performed on multiple feature types to compare effectiveness.

### 4. Visualization

- 2D and 3D scatter plots were generated for PCA outputs.
- Interactive 3D visualizations helped analyze cluster separation and quality.

### 5. Optimization Techniques

- Batch processing (1000 images at a time) was used to handle memory efficiently.
- Fixed random state ensured reproducibility.
- Feature caching avoided recomputation during repeated experiments.

## Challenges and Solutions

- **Clustering Quality Validation:** Used Silhouette Score for determining PCA components and cluster effectiveness.

## Code Instructions to Run

1. **Install Dependencies:**

   Ensure the following libraries are installed:
   ```bash
   pip install numpy scipy scikit-learn matplotlib
