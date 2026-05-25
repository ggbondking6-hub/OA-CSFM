Joint Modeling of Visibility and Reliability for Occlusion-Robust Dynamic Facial Expression Recognition

[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0+-EE4C2C.svg)](https://pytorch.org/)

This repository provides the complete source code, training scripts, and configuration pipelines necessary to fully reproduce the experiments and results presented in our paper. We propose an Occlusion-Aware Collaborative Spatio-temporal Fusion Method (OA-CSFM) framework to address occlusion-induced degradation and temporal reliability mismatch in dynamic facial expression recognition (DFER).

---

1. Data Availability Statement

The DFEW dataset analyzed during the current study is available at https://dfew-dataset.github.io/.

The FERV39k dataset analyzed during the current study is available at https://wangy3dkx.github.io/


2. Environment Setup
All experiments were conducted using PyTorch and Linux/Windows environments. To set up the exact reproduction environment for OA-CSFM, please execute the following commands:

```bash
Clone the repository and navigate to the root directory
cd OA-CSFM_Code

Create and activate a conda environment
conda create -n oacsfm_env python=3.8 -y
conda activate oacsfm_env

Install core dependencies (Adjust CUDA version according to your hardware, e.g., cu118)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install numpy scikit-learn