# WM811k-Silicon-Wafer-Map-Dataset-and-Defect-Detection-Using-CNNs-with-Grad-CAM

This repository provides a curated **WM811k silicon wafer map dataset** subset and MATLAB-based CNN + Grad-CAM implementation for wafer defect detection. The work is based on:

**[Enhancing Defect Recognition: Convolutional Neural Networks for Silicon Wafer Map Analysis](https://ieeexplore.ieee.org/document/10561853)**  
Published in the 2024 3rd International Conference on Advancement in Electrical and Electronic Engineering (ICAEEE).

![image](https://github.com/user-attachments/assets/95844223-9801-4472-b921-efab63e1350f)

---

## Dataset Summary

- 902 images across 9 defect classes: Center, Donut, Edge Local, Edge Ring, Local, Near Full, None, Random, Scratch
- Images are resized to **32x32 pixels**
- Stored in `WM811k_Dataset.rar` and expected to be extracted into class-named folders

---

## Developer Setup

### Prerequisites

- MATLAB with Deep Learning Toolbox (required for `trainNetwork`, `imageDatastore`, and `gradCAM`)
- Access to the dataset archive (`WM811k_Dataset.rar`)

### Extract the Dataset

Extract `WM811k_Dataset.rar` into a folder that contains the class subfolders (for example `Dataset/Center`, `Dataset/Donut`, etc.). The MATLAB script uses folder names as labels.

### MATLAB Workflow (`grad.m`)

1. Open `grad.m` in MATLAB.
2. Update the dataset path in the `imageDatastore` call to your extracted dataset directory.
3. Run the script. It will:
   - Split data into train/validation sets
   - Train the CNN
   - Evaluate accuracy and confusion matrix
   - Compute precision/recall/F1
   - Generate a Grad-CAM visualization (update the sample image path if needed)

### Notebook Workflow (`defect-detection-using-cnns-and-gradcam-vis.ipynb`)

The notebook contains MATLAB-style code cells. Open it in a Jupyter environment configured for MATLAB, or copy the cells into MATLAB for execution.

---

## Repository Structure

```
.
├── WM811k_Dataset.rar
├── grad.m
├── defect-detection-using-cnns-and-gradcam-vis.ipynb
└── README.md
```

After extraction, your dataset directory should look like:

```
Dataset/
├── Center/
├── Donut/
├── Edge_Local/
├── Edge_Ring/
├── Local/
├── Near_Full/
├── None/
├── Random/
└── Scratch/
```

---

## Citation

If you use this dataset or code, please cite:

```
@inproceedings{junayed2024enhancing,
  title={Enhancing Defect Recognition: Convolutional Neural Networks for Silicon Wafer Map Analysis},
  author={Muhammad Junayed, Tanzeem Tahmeed Reza, Md. Saiful Islam},
  booktitle={2024 3rd International Conference on Advancement in Electrical and Electronic Engineering (ICAEEE)},
  year={2024},
  organization={IEEE}
}
```

---

## Related Paper

- [Enhancing_Defect_Recognition_Convolutional_Neural_Networks_for_Silicon_Wafer_Map_Analysis.pdf](https://github.com/user-attachments/files/18211773/Enhancing_Defect_Recognition_Convolutional_Neural_Networks_for_Silicon_Wafer_Map_Analysis.pdf)
