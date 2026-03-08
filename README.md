# Object Recognition System

An **Object Recognition System** that combines **deep learning features from ResNet** with **traditional computer vision techniques** such as **image moments and handcrafted geometric features** to recognize objects in images and video streams.

The system supports **interactive training and inference modes**, allowing users to toggle between **ResNet-based semantic features** and **manually constructed features** for object recognition.

---

# Demo Videos

### Training
[Video](https://drive.google.com/file/d/1RmhAkGk_8KUN-JrcxCYwXj8BVKKEFbjf/view?usp=sharing) 

### Inference
[Video](https://drive.google.com/file/d/1-JsssRtIQRh4OQ7zdTvtP-G_-N7a9Rfd/view?usp=sharing)

---

# Features

## Handcrafted Features

| Feature Type | Description | Dimensions |
|--------------|-------------|------------|
| **Percent Filled** | Ratio of region pixels to bounding box area | 1-d |
| **Aspect Ratio** | Height / Width of oriented bounding box | 1-d |
| **Hu Moments** | 7 rotation and scale invariant moments (log-transformed) | 7-d |

**Total Feature Size:** 9-d  

**Distance Metric:**  
Scaled Euclidean distance  

\[
(x_1 - x_2) / \sigma
\]

---

## Deep Learning Features

| Feature Type | Description | Dimensions |
|--------------|-------------|------------|
| **ResNet-18** | Semantic embedding extracted from pre-trained ResNet-18 (ImageNet weights) | 512-d |

**Distance Metric:** Cosine similarity

---

# System Pipeline

## Preprocessing

| Step | Description |
|-----|-------------|
| **Gaussian Blur** | Smooths image before thresholding |
| **Thresholding** | Converts input image to binary |
| **Connected Components** | Labels and segments individual regions |
| **Area Filtering** | Removes small noise regions (< minArea) |
| **Label Remapping** | Reassigns valid regions to consecutive IDs |

---

## Feature Extraction

| Step | Description |
|-----|-------------|
| **Moment Computation** | Computes raw moments (m00, m10, m01, m20, m02, m11) |
| **Central Moments** | Computes mu20, mu02, mu11 |
| **Orientation** | Computes primary axis angle using second central moments |
| **Oriented Bounding Box** | Finds min/max projections along eigenvector axes |
| **Hu Moments** | Computes 7 rotation-invariant log-transformed moments |
| **ResNet Embedding** | Region is extracted, rotated, and passed through ResNet-18 to obtain a 512-d feature vector |

---

# Classification

| Method | Description |
|------|-------------|
| **Nearest Neighbor** | Compares query features against stored database |
| **Scaled Euclidean** | Used for handcrafted features (normalized by standard deviation) |
| **Cosine Distance** | Used for ResNet embeddings |
| **Unknown Detection** | Bounding boxes are not drawn if object distance exceeds a threshold |

---

# GUI Layout

## Control Panel

| Button | Function |
|------|-----------|
| **Train Off / Train On** | Toggle between training and inference mode |
| **ResNet Off / ResNet On** | Toggle between handcrafted and ResNet features |
| **Next** | Advance to next training stage |
| **Quit** | Exit the application |

A **status indicator (green text)** displays the current system mode.

---

# Training Stages

| Stage | Mode | Description |
|------|------|-------------|
| **1** | Threshold | Applies thresholding to the input image |
| **2** | Segmentation | Applies connected components segmentation. User selects region ID via dialog |
| **3** | Extract Features | Computes handcrafted features or ResNet embedding |
| **4** | Store Features | User enters object name and features are stored in CSV file |

Separate CSV files are maintained for **handcrafted features** and **ResNet embeddings**.

The **Next** button moves the system through each stage.

---

# Inference Mode

During inference, the system performs the following steps:

1. Thresholds and segments the input image or video frame
2. Extracts features for each valid region
3. Compares features with the stored object database
4. Draws oriented bounding boxes around recognized objects
5. Displays predicted object labels
6. Labels objects as **"Unknown"** if the distance exceeds a threshold

---

# Distance Metrics

| Metric | Used For |
|------|-----------|
| **Scaled Euclidean** | Handcrafted features (9-d) |
| **Cosine Distance** | ResNet-18 embeddings (512-d) |

---

# Extensions

Additional improvements implemented in this project:

1. **Interactive GUI** built using OpenCV callbacks  
2. **Expanded object database** including the following objects:
   - Pen
   - Pestle
   - Screwdriver
   - Mouse
   - Candy
   - Perfume
   - Scissors
   - Spoon
   - Phone
3. **2D visualization of embeddings**

---

# Deadline Extension

Approval was granted by the professor for **late submission due to health reasons**, therefore the **time travel day policy does not apply**.

---

# Requirements

- **OpenCV 4.x**
- **C++17 or later**
- **CMake 3.10+**
- **Pre-trained ResNet-18 ONNX model**

---

# Building

### Windows (Visual Studio)

```bash
mkdir build
cd build
cmake --build build --clean-first --config Debug