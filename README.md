# Enhancing Real-Time 3D Object Detection for Autonomous Driving via CBAM-FPN-ResNet18

This repository contains the code, results, and model enhancements from our research project on 3D object detection using only LiDAR point clouds.  
**"Enhancing Real-Time 3D Object Detection for Autonomous Driving via CBAM-FPN-ResNet18"**

## 📌 Overview

This project explores LiDAR-based 3D object detection with a focus on real-time, efficient architectures suitable for deployment in autonomous vehicles. We:

- Compared **VoxelNet** and **SFA3D** on the KITTI dataset
- Selected **SFA3D** as the baseline due to its real-time capabilities
- Proposed an enhanced model: **CBAM-FPN-ResNet18**
- Improved detection of **occluded** and **small-scale** objects through attention and multi-scale fusion

---

## 📚 Table of Contents

- [1. Introduction](#1-introduction)
- [2. Baselines](#2-baselines)
  - [2.1 VoxelNet](#21-voxelnet)
  - [2.2 SFA3D](#22-sfa3d)
- [3. Enhancement: CBAM-FPN-ResNet18](#3-enhancement-cbam-fpn-resnet18)
- [4. Qualitative Results](#4-qualitative-results)
- [5. Quantitative Comparison](#5-quantitative-comparison)
- [6. Setup Instructions](#6-setup-instructions)
- [7. Citations](#7-citations)
- [8. License](#8-license)

---

## 1. Introduction

Accurate 3D object detection is crucial for autonomous navigation, but real-time constraints demand lightweight models. We study voxel-based and BEV-based approaches, ultimately enhancing the SFA3D framework with channel and spatial attention modules (CBAM) and multi-scale feature fusion.

---

## 2. Baselines

### 2.1 VoxelNet

VoxelNet voxelizes raw LiDAR input and uses 3D convolutions to learn features for 3D bounding box regression.

- **Framework:** PyTorch (based on [qianguih/voxelnet](https://github.com/qianguih/voxelnet))
- **Training time:** ~7 days (55 epochs)
- **Drawbacks:** High memory consumption, slow inference (~350ms/frame), unstable convergence

📈 **Representative Outputs:**

![VoxelNet Loss](https://github.com/user-attachments/assets/8feda395-a8ed-4b5c-b105-f4391bc62364)  
**Figure 1 – VoxelNet Training Loss Curve:**  
This figure illustrates the unstable convergence behavior of VoxelNet. The training loss remained high (above 40) for a significant duration, with frequent oscillations and spikes. This instability reflects difficulties in optimizing the voxel-based architecture on sparse LiDAR data with fixed voxel resolution.

![VoxelNet Output](https://github.com/user-attachments/assets/4a4c42e1-2412-4817-b3b6-c744becc0025)  
**Figure 2 – VoxelNet Detection Output:**  
Shown here are Bird’s Eye View (BEV) and 3D real-world views. VoxelNet often misses distant or partially occluded objects. This is attributed to:
- Lack of global context modeling
- Fixed voxel sizing limiting resolution
- High computational load and slower convergence
- Additionally, the model tends to hallucinate and generate false positive detections, placing 3D bounding boxes where no real objects exist. This behavior indicates insufficient learning and highlights the need for more extensive training or better regularization techniques.
---

### 2.2 SFA3D

SFA3D is a single-stage, BEV-based detector using a ResNet18 Keypoint Feature Pyramid Network (KFPN). It predicts 7-DOF bounding boxes with multiple regression heads.

- **Training time:** ~9 hours (120 epochs)
- **Inference speed:** Real-time
- **Limitations:** Difficulties with small/occluded objects

📈 **Representative Outputs:**

![SFA3D BEV](https://github.com/user-attachments/assets/a3cac736-ef5e-409b-920a-dcc50abb8fb0)  
![SFA3D Output](https://github.com/user-attachments/assets/245f6b7a-738b-44ac-8b7d-48e801e4e72c)

---

## 3. Enhancement: CBAM-FPN-ResNet18

We enhance SFA3D by introducing:

- **CBAM (Convolutional Block Attention Module):** Focuses on "what" and "where" in sparse LiDAR scenes
- **FPN (Feature Pyramid Network):** Enables multi-scale fusion across different resolution levels
- **Improved generalization:** Better localization in cluttered, occluded, or long-range scenes

🛠️ **Implementation Highlights:**
- CBAM inserted after ResNet18's mid-level residual blocks
- Softmax-weighted fusion across three feature pyramid levels
- Retains all original prediction heads

---

## 4. Qualitative Results

We compare the qualitative performance of:

- 🔵 **SFA3D Baseline**
- 🟢 **CBAM-SFA3D (Ours)**

### Case 1: Occluded Pedestrian  
_Left: CBAM-SFA3D successfully detects a pedestrian partially occluded behind a vehicle. Right: The baseline SFA3D fails to detect the same pedestrian. The missed region is highlighted in the bottom-right of the image for clarity._

![Occlusion](https://github.com/user-attachments/assets/2e76139e-3486-4271-a5ad-c465c117216d)

### Case 2: Long-range Small Object  
_Left: CBAM-SFA3D detects a small pedestrian at long range in a cluttered urban scene. Right: SFA3D baseline fails to localize the pedestrian. The missed detection is indicated in the bottom-right of the image._
![Long-range](https://github.com/user-attachments/assets/6457509f-1c1e-4f56-8b4a-1b4498b5a263)

---

## 5. Quantitative Comparison

Here’s a comparison of **3D Average Precision (AP)** on KITTI validation:

| Class      | IoU | Difficulty | SFA3D (%) | CBAM-SFA3D (%) |
|------------|-----|------------|-----------|----------------|
| **Car**    | 0.7 | Moderate   | 87.74     | 64.43          |
| **Car**    | 0.5 | Moderate   | 90.77     | 88.97          |
| **Pedestrian** | 0.5 | Moderate | 44.97 | 24.43 |
| **Cyclist** | 0.5 | Moderate | 30.56 | 19.39 |

⚠️ *While CBAM-SFA3D underperforms under strict metrics, it excels in generalization and qualitative scenarios.*

---

## 6. Setup Instructions

```bash
# Clone repo
git clone https://github.com/ChinmayAmrutkar/3D-Object_Detection.git
cd 3D-Object_Detection

# Install dependencies (inside virtualenv or Conda)
pip install -r requirements.txt

# Train (example)
python train.py --cfg configs/cbam_resnet18.yaml

# Inference (example)
python evaluate.py --ckpt outputs/cbam_model.pth
```

---

## 7. Citations

### 📘 SFA3D
```bibtex
@misc{maudzung2020sfa3d,
  author = {Nguyen Mau Dung},
  title = {Super-Fast-Accurate-3D-Object-Detection-PyTorch},
  year = {2020},
  howpublished = {\url{https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection}}
}
```

### 📗 VoxelNet
```bibtex
@inproceedings{zhou2018voxelnet,
  title={VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection},
  author={Zhou, Yin and Tuzel, Oncel},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4490--4499},
  year={2018}
}
```

### 💻 VoxelNet GitHub Implementation
```bibtex
@misc{qianguih_voxelnet,
  author = {Qiangui Huang},
  title = {VoxelNet PyTorch Implementation},
  howpublished = {\url{https://github.com/qianguih/voxelnet}},
  year = {2019}
}
```

### 🧠 CBAM-SFA3D (Ours)
```bibtex
@misc{amrutkar2024cbamsfa3d,
  author = {Chinmay Amrutkar and Jnana Venkata Subhash Boyapati and Swaraj Akurathi and Thet Htar Wai},
  title = {Enhancing Real-Time 3D Object Detection for Autonomous Driving via CBAM-FPN-ResNet18},
  howpublished = {\url{https://github.com/ChinmayAmrutkar/Enhancing Real-Time 3D Object Detection for Autonomous Driving via CBAM-FPN-ResNet18}},
  year = {2024}
}
```

---

## 8. License

This project is released under the [MIT License](LICENSE).  
© 2024 Chinmay Amrutkar, Arizona State University

---

## 📄 Additional Information

For full methodology, results, architectural diagrams, and future directions, please refer to our detailed report:  
**[Enhancing Real-Time 3D Object Detection for Autonomous Driving via CBAM-FPN-ResNet18.pdf]([./Advanced_3D_Object_Detection_for_Autonomous_Driving.pdf](https://github.com/ChinmayAmrutkar/Enhancing-Real-Time-3D-Object-Detection-for-Autonomous-Driving-via-CBAM-FPN-ResNet18/blob/main/Enhancing%20Real-Time%203D%20Object%20Detection%20for%20Autonomous%20Driving%20via%20CBAM-FPN-ResNet18.pdf))**

---

## 🛠 Authors

- Chinmay Ravindra Amrutkar ([chinmay.amrutkar@asu.edu](mailto:chinmay.amrutkar@asu.edu))
- Jnana Venkata Subhash Boyapati  
- Swaraj Akurathi  
- Thet Htar Wai

---
