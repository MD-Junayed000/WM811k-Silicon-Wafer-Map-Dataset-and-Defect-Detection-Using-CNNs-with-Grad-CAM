# WM811k-Silicon-Wafer-Map-Dataset-and-Defect-Detection-Using-CNNs-with-Grad-CAM

This repository provides the **WM811k silicon wafer map dataset** subset, MATLAB implementation, and Jupyter Notebook(code in MATLAB software) for defect detection in semiconductor wafers using Convolutional Neural Networks (CNNs). The research is based on the paper:

**[Enhancing Defect Recognition: Convolutional Neural Networks for Silicon Wafer Map Analysis]([https://doi.org/your-paper-link](https://doi.org/10.1109/ICAEEE62219.2024.10561853))**  
Published in the 2024 3rd International Conference on Advancement in Electrical and Electronic Engineering (ICAEEE).

---

## Features

1. **Dataset**:  
   - A curated subset of the WM811k silicon wafer map dataset with 902 images categorized into 9 defect classes:
     - Center
     - Donut
     - Edge Local
     - Edge Ring
     - Local
     - Near Full
     - None
     - Random
     - Scratch

   - Images are resized to **32x32 pixels** for efficient processing.

2. **MATLAB Code (`grad.m`)**:  
   - Implements a CNN architecture with five convolutional layers, batch normalization, ReLU activation, and Grad-CAM visualization.
     **MATLAB Implementation¶**
     >> Run Code in MATLAB to:
          >Train the CNN

          >valuate its performance

         > Visualize Grad-CAM results


3. **Jupyter Notebook (`defect-detection-using-cnns-and-gradcam-vis.ipynb`)**:  
   - Provides an interactive Python-based implementation of defect detection and Grad-CAM visualization.
   - Useful for researchers who prefer Python-based workflows.

4. **Research Paper**:  
   - [Enhancing_Defect_Recognition_Convolutional_Neural_Networks_for_Silicon_Wafer_Map_Analysis.pdf](https://github.com/user-attachments/files/18211773/Enhancing_Defect_Recognition_Convolutional_Neural_Networks_for_Silicon_Wafer_Map_Analysis.pdf)


---
## Repository Structure
WM811k-Defect-Detection/
├── Dataset/
│   ├── Center/
│   ├── Donut/
│   ├── Edge_Local/
│   ├── Edge_Ring/
│   ├── Local/
│   ├── Near_Full/
│   ├── None/
│   ├── Random/
│   └── Scratch/
├── grad.m
├── defect-detection-using-cnns-and-gradcam-vis.ipynb
├── README.md
└── Paper/
    └── Enhancing_Defect_Recognition_CNN.pdf

