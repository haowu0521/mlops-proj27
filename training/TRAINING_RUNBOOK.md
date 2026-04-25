# Training Runbook

## What this folder is for
This folder owns the summarization training and retraining workflow.

Official entry points:
- `train.py`: normal summarization training
- `prepare_retraining_dataset_from_api.py`: build retraining dataset from `/reviews`
- `run_retraining_from_reviews.py`: end-to-end API-driven retraining

## Environment variables
Required in most cases:
- `DATA_API_BASE`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Optional:
- `MIN_RETRAIN_EXAMPLES`
- `TRAIN_CONFIG_PATH`
- `MLFLOW_REGISTERED_MODEL_NAME`
- `MLFLOW_REGISTERED_MODEL_ALIAS`

Storage-safety knobs:
- `TRAIN_SAVE_STRATEGY=no`
- `MLFLOW_LOG_HF_MODEL_FILES=0`
- `CLEAN_LOCAL_OUTPUT_DIR_AFTER_REGISTER=1`

## Normal training
```bash
cd ~/mlops-proj27/training

export MLFLOW_TRACKING_URI="http://129.114.27.10:30500"
export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:30900"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
export TRAIN_SAVE_STRATEGY="no"
unset MLFLOW_LOG_HF_MODEL_FILES

python3 train.py --config config.yaml
```

## Retraining from API reviews
```bash
cd ~/mlops-proj27/training

export DATA_API_BASE="http://129.114.27.10:30800"
export MLFLOW_TRACKING_URI="http://129.114.27.10:30500"
export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:30900"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
export MIN_RETRAIN_EXAMPLES="1"
export TRAIN_SAVE_STRATEGY="no"
unset MLFLOW_LOG_HF_MODEL_FILES

python3 prepare_retraining_dataset_from_api.py --write-empty
cat data/retraining_stats.json
python3 run_retraining_from_reviews.py
```

## How to tell it succeeded
- `data/retraining_stats.json` exists
- `eligible_examples > 0`
- training finishes without error
- MLflow shows a new registered model version
