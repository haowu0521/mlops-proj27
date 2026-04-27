#!/bin/bash
set -e

cd /home/cc/mlops-proj27/training

mkdir -p data outputs logs

echo "========== Retraining started at $(date) =========="

sudo docker run --rm \
  -e DATA_API_BASE="http://129.114.27.10:30800" \
  -e RETRAIN_DATA_DIR="/app/training/data" \
  -e MIN_RETRAIN_EXAMPLES="1" \
  -e MEETING_IDS="cb0f7be0-e418-4258-880f-0048c52b3c59,369f286d-a70f-48e1-9f7d-fe5487b6e727,d7b68cba-c06d-484e-8046-226c1511d654,dc137b0c-374f-4bdb-a6c7-0ab7b54bbdfb,5325d089-528e-4777-9bf3-f0ae93c42f10,769ba28f-da29-4f65-8267-5398a6b4821f" \
  -v "$PWD/data:/app/training/data" \
  -v "$PWD/outputs:/app/training/outputs" \
  proj27-training \
  python3 run_retraining_from_reviews_v2.py

echo "========== Retraining finished at $(date) =========="
