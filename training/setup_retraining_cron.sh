#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$PROJECT_DIR/data" "$PROJECT_DIR/outputs" "$PROJECT_DIR/logs"

cat > "$PROJECT_DIR/run_retraining_docker.sh" <<EOF
#!/bin/bash
set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

PROJECT_DIR="$PROJECT_DIR"

cd "\$PROJECT_DIR"

mkdir -p data outputs logs

echo "========== Retraining check started at \$(date) =========="

if docker info >/dev/null 2>&1; then
  docker run --rm \
    -e DATA_API_BASE="http://129.114.27.10:30800" \
    -e RETRAIN_DATA_DIR="/app/training/data" \
    -v "\$PWD/data:/app/training/data" \
    -v "\$PWD/outputs:/app/training/outputs" \
    proj27-training \
    python3 run_retraining_from_reviews_v2.py
else
  sudo -n docker run --rm \
    -e DATA_API_BASE="http://129.114.27.10:30800" \
    -e RETRAIN_DATA_DIR="/app/training/data" \
    -v "\$PWD/data:/app/training/data" \
    -v "\$PWD/outputs:/app/training/outputs" \
    proj27-training \
    python3 run_retraining_from_reviews_v2.py
fi

echo "========== Retraining check finished at \$(date) =========="
EOF

chmod +x "$PROJECT_DIR/run_retraining_docker.sh"

CRON_LINE="*/30 * * * * $PROJECT_DIR/run_retraining_docker.sh >> $PROJECT_DIR/logs/retraining_cron.log 2>&1"

# Remove old retraining cron line if it already exists, then add the new one
( crontab -l 2>/dev/null | grep -v "$PROJECT_DIR/run_retraining_docker.sh" ; echo "$CRON_LINE" ) | crontab -

echo "[OK] Automatic retraining cron job installed."
echo "[OK] It will run every 30 minutes."
echo "[OK] Log file: $PROJECT_DIR/logs/retraining_cron.log"
echo
echo "Current cron jobs:"
crontab -l
