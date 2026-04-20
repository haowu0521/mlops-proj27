#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="python3"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://129.114.26.182:30500}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://129.114.26.182:30900}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minio}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minio123}"
export MLFLOW_REGISTERED_MODEL_NAME="${MLFLOW_REGISTERED_MODEL_NAME:-jitsi-summarizer}"

INPUT_FILE="data/feedback_records.jsonl"
TRAIN_OUTPUT="data/retraining_ready_train.jsonl"
VAL_OUTPUT="data/retraining_ready_val.jsonl"
CONFIG_FILE="config.yaml"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file not found: $INPUT_FILE"
  exit 1
fi

echo "Starting retraining pipeline..."
echo "Working directory: $SCRIPT_DIR"
echo "Using Python: $PYTHON_BIN"
echo "Input feedback file: $INPUT_FILE"

$PYTHON_BIN prepare_retraining_dataset.py \
  --input "$INPUT_FILE" \
  --train-output "$TRAIN_OUTPUT" \
  --val-output "$VAL_OUTPUT"

echo "Dataset preparation finished."
echo "Train file: $TRAIN_OUTPUT"
echo "Val file: $VAL_OUTPUT"

$PYTHON_BIN train.py --config "$CONFIG_FILE"

echo "Retraining pipeline finished successfully."
