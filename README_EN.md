[简体中文](README.md) | [English](README_EN.md)
# YOLOv12: Streamlined & Pure PyTorch Implementation
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

## Overview
This project is a pure PyTorch reimplementation of YOLOv12. The architecture is decoupled and readable, featuring an **original flow chart** of the YOLOv12 model.<br> 
This repository provides a decoupled training pipeline with automated dataset processing tools, enabling rapid deployment and flexible configuration. The implementation achieves 91.5% mAP@0.5 and 62.4% mAP@0.5:0.95 (Large scale) on the BCCD dataset.

## Model Structure
<details>
<summary><b>Click to expand the model structure</b></summary>
  <p align="center">
      <img src="model_data/model_structure.svg" width="97%" >
    <br>
  </p>
  
</details>

## Table of Contents
1. [Overview](#overview)
2. [Model Structure](#model-structure)
3. [Performance](#performance)
4. [Detection Results](#detection-results)
5. [Download](#download)
6. [Environment](#environment)
7. [Quick Start](#quick-start)
8. [Reference](#reference)

---

## Performance
Comparison of YOLOv12 (n, s, m, l scales) trained on the BCCD dataset.

<img src="model_data/model_metrics.svg" width="90%" >

<details>
<summary><b>Click for detailed metrics</b></summary>

<details>
<summary><b>Nano (n)</b></summary>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505<br>
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.890<br>
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.531<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.376<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.371<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.578<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.542<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.615<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.525<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.596<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638<br>
</details>

<details>
<summary><b>Small (s)</b></summary>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583<br>
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.885<br>
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.645<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.458<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.620<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.694<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.525<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.647<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.742<br>
</details>

<details>
<summary><b>Medium (m)</b></summary>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.600<br>
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.893<br>
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.675<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.289<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.371<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.622<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.697<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.525<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.645<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.741<br>
</details>

<details>
<summary><b>Large (l)</b></summary>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.624<br>
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.915<br>
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.681<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.515<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453<br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.719<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.385<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.644<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.714<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.550<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.654<br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773<br>
</details>

</details>

## Detection Results
<p align="center">
  <img src="model_data/model_output.png" width="60%" >
  <br>
  <i>yolov12_l</i>
</p>

## Download
### Dataset
[BCCD Dataset (VOC)](https://github.com/Shenggan/BCCD_Dataset)<br>

### Pretrained Weights
| Model | Dataset | Size | mAP<sup>val<br>0.5 | mAP<sup>val<br>0.5:0.95 | Download |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv12-n** | BCCD | 640x640 | 89.0% | 50.5% |[yolov12_n.pth](https://github.com/zoomzonezero/yolov12-pure-pytorch/releases/download/pretrained-weights/yolov12_n.pth) |
| **YOLOv12-s** | BCCD | 640x640 | 88.5% | 58.3% |[yolov12_s.pth](https://github.com/zoomzonezero/yolov12-pure-pytorch/releases/download/pretrained-weights/yolov12_s.pth) |
| **YOLOv12-m** | BCCD | 640x640 | 89.3% | 60.0% |[yolov12_m.pth](https://github.com/zoomzonezero/yolov12-pure-pytorch/releases/download/pretrained-weights/yolov12_m.pth) |
| **YOLOv12-l** | BCCD | 640x640 | 91.5% | 62.4% |[yolov12_l.pth](https://github.com/zoomzonezero/yolov12-pure-pytorch/releases/download/pretrained-weights/yolov12_l.pth) |

## Environment
- Python: 3.10+ <br>
- PyTorch: 2.0.0+ <br>
- CUDA: 11.8+ <br>

## Quick Start
1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/zoomzonezero/yolov12-pure-pytorch.git
cd yolov12-pure-pytorch

# Install dependencies
pip install -r requirements.txt
```

2. Prediction
Download the pre-trained weights.(Refer to **[Download](#download)**)<br>
Update the weight path and model scale in `config.py`.<br>
Run the prediction script:
```bash
python predict.py
# Then input your image path when prompted
```

3. Training
Prepare your dataset and ensure v_class.txt is updated accordingly.<br>
Configure your hyperparameters in `config.py`.<br>
Start training:
```bash
# Standard training
python train.py

# Linux automated training (via job.txt)
bash train_linux/run.sh
```
### More Details
For advanced configuration, dataset preparation, and troubleshooting, please refer to the detailed **[Chinese Documentation](README.md)**.

## Reference
[YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)<br>
[https://github.com/sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)<br>
[https://github.com/bubbliiiing/yolox-pytorch](https://github.com/bubbliiiing/yolox-pytorch)<br>