# DL Classification API

## Overview
Train a small CNN on CIFAR-10 and serve via FastAPI /predict endpoint. Works on:
- NVIDIA GPU (Linux/WSL with CUDA) — use GPU-enabled TF and Docker GPU image
- macOS Apple Silicon (M1/M2/M4) — use `tensorflow-macos` + `tensorflow-metal` (run natively, not Docker)

## Setup (Apple Silicon - native)
1. Create venv:
   python3 -m venv venv && source venv/bin/activate
2. Upgrade pip:
   pip install --upgrade pip
3. Install mac tensorflow packages:
   pip install -r requirements_mac_m1_m2_m4.txt
4. Train:
   EPOCHS=10 python train.py
   (model saved to ./saved_model)
5. Run server:
   python app.py
6. Predict:
   curl -F "file=@my_image.jpg" http://127.0.0.1:8000/predict

## Setup (NVIDIA GPU - Linux)
1. Install NVIDIA drivers, CUDA, cuDNN (follow NVIDIA & TF compatibility).
2. Create venv: python3 -m venv venv && source venv/bin/activate
3. pip install -r requirements_gpu.txt
4. Train on GPU:
   EPOCHS=10 python train.py
5. Run server:
   python app.py
6. Or build GPU docker:
   docker build -t dl-classifier:gpu -f Dockerfile.gpu .
   docker run --gpus all -p 8000:8000 dl-classifier:gpu

## CPU Docker (if no GPU):
docker build -t dl-classifier:cpu -f Dockerfile.cpu .
docker run -p 8000:8000 dl-classifier:cpu

## Notes
- On mac M-series prefer native run; running TF with GPU in Docker on mac is experimental.
- For production: add batching, authentication, logging, monitoring, and model versioning.
