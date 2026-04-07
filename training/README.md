# Jitsi Summarization Training Pipeline

This repository contains the training side of our course project: post meeting summarization for Jitsi. The goal is to train and compare candidate summarization models, track runs in MLflow, and provide reproducible training artifacts that can be executed on Chameleon inside a Docker container.

## Files

- `train.py`: main training script
- `requirements.txt`: Python dependencies
- `Dockerfile`: training container definition
- `config.yaml`: baseline configuration
- `config_v1.yaml`: candidate v1 configuration
- `config_v2.yaml`: candidate v2 configuration

## MLflow

- Tracking URI: `http://129.114.26.89:8000`
- Experiment name: `jitsi-summarization-v2`

## Training command

Baseline:
```bash
sudo docker run --rm -w /app -v ~/summarization_training/training:/app proj27-sumtrain:baseline python -u train.py --config config.yaml
