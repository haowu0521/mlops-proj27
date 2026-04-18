#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -d ".venv" ]]; then
  source .venv/bin/activate
else
  echo "Missing .venv in $SCRIPT_DIR"
  exit 1
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://129.114.26.182:30500}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://129.114.26.182:30900}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minio}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minio123}"
export MLFLOW_REGISTERED_MODEL_NAME="${MLFLOW_REGISTERED_MODEL_NAME:-jitsi-summarizer}"

INPUT_FEEDBACK="${1:-data/feedback_records.jsonl}"
TRAIN_OUTPUT="data/retraining_ready_train.jsonl"
VAL_OUTPUT="data/retraining_ready_val.jsonl"
BASE_CONFIG="${2:-config.yaml}"
GENERATED_CONFIG="config.retraining.generated.yaml"

if [[ ! -f "$INPUT_FEEDBACK" ]]; then
  echo "Input feedback file not found: $INPUT_FEEDBACK"
  exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config file not found: $BASE_CONFIG"
  exit 1
fi

echo "Step 1/3: Preparing retraining dataset..."
python prepare_retraining_dataset.py \
  --input "$INPUT_FEEDBACK" \
  --train-output "$TRAIN_OUTPUT" \
  --val-output "$VAL_OUTPUT"

if [[ ! -f "$TRAIN_OUTPUT" ]]; then
  echo "Training output file was not created: $TRAIN_OUTPUT"
  exit 1
fi

TRAIN_LINES=$(wc -l < "$TRAIN_OUTPUT" | tr -d ' ')
VAL_LINES=0
if [[ -f "$VAL_OUTPUT" ]]; then
  VAL_LINES=$(wc -l < "$VAL_OUTPUT" | tr -d ' ')
fi

echo "Train samples: $TRAIN_LINES"
echo "Validation samples: $VAL_LINES"

if [[ "$TRAIN_LINES" -eq 0 ]]; then
  echo "No retraining samples generated. Stopping."
  exit 1
fi

echo "Step 2/3: Generating retraining config..."
python - "$BASE_CONFIG" "$GENERATED_CONFIG" "$TRAIN_OUTPUT" "$VAL_OUTPUT" "$VAL_LINES" <<'PY'
import sys
import yaml

base_config, output_config, train_file, val_file, val_lines = sys.argv[1:6]
val_lines = int(val_lines)

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})
cfg["data"]["dataset_name"] = None
cfg["data"]["dataset_config"] = None
cfg["data"]["train_file"] = train_file
cfg["data"]["test_file"] = None
cfg["data"]["text_column"] = "input_transcript"
cfg["data"]["summary_column"] = "target_summary"
cfg["data"]["make_test_from_validation"] = False

if val_lines > 0:
    cfg["data"]["validation_file"] = val_file
else:
    cfg["data"]["validation_file"] = None
    cfg["data"]["validation_split"] = 0.2

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"Generated config: {output_config}")
print(f"Validation file used: {cfg['data']['validation_file']}")
PY

echo "Step 3/3: Starting retraining..."
python train.py --config "$GENERATED_CONFIG"

echo "Retraining pipeline completed successfully."
